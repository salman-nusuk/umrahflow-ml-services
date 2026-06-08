"""FastAPI sidecar that exposes the LangGraph agent.

POST /turn  { "conversation_id": "...", "message_id": "...", "text": "..." }
GET  /healthz

A background drain thread polls Postgres every POLL_S seconds for conversations
that have unread inbound messages quiet for at least DEBOUNCE_S — when one is
found it processes the entire burst as a single LangGraph turn. This collapses
WhatsApp 5-image albums (5 separate webhooks) into one consolidated reply.
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import concurrent.futures
import contextlib
import hashlib
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# load project-root .env (one level up from this service folder)


from agent import build_agent
from tools import _conn, reset_send_count  # reuse helpers

# Sentry — no-op when SENTRY_DSN_AGENT (or fallback SENTRY_DSN) is unset.
_SENTRY_DSN = os.environ.get("SENTRY_DSN_AGENT") or os.environ.get("SENTRY_DSN")
_sentry_sdk = None
if _SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            traces_sample_rate=0.1,
            environment=os.environ.get("SENTRY_ENV", "production"),
        )
        _sentry_sdk = sentry_sdk
    except ImportError:
        pass


def _capture(e: Exception) -> None:
    """Report to Sentry when configured; silent otherwise."""
    if _sentry_sdk is not None:
        try:
            _sentry_sdk.capture_exception(e)
        except Exception:
            pass


def _capture_message(msg: str, level: str = "warning") -> None:
    if _sentry_sdk is not None:
        try:
            _sentry_sdk.capture_message(msg, level=level)
        except Exception:
            pass


agent = build_agent()
app = FastAPI(title="Safr-e-Ibadat Agent")


# Drain heartbeat — /healthz uses this to detect a dead drain thread.
_PROCESS_STARTED_AT: float = time.time()
_drain_last_tick: float = 0.0
_drain_loop_started: bool = False


@app.get("/healthz")
def healthz():
    """Liveness probe. Returns 503 when the drain loop is dead or never started.

    Without this, ops only learns the drain is wedged when customers stop
    getting replies. The heartbeat is updated each iteration of `_drain_loop`.
    """
    now = time.time()
    age = now - _drain_last_tick if _drain_last_tick else None
    uptime = now - _PROCESS_STARTED_AT
    drain_disabled = os.environ.get("AGENT_DRAIN_DISABLED") == "1"

    body = {
        "ok": True,
        "model": "claude-sonnet-4-6",
        "drain_last_tick_age_s": age,
        "drain_loop_started": _drain_loop_started,
        "uptime_s": uptime,
    }

    if drain_disabled:
        return body

    # never started after 60s of uptime → unhealthy
    if not _drain_loop_started and uptime > 60:
        body["ok"] = False
        body["reason"] = "drain_loop_never_started"
        return JSONResponse(status_code=503, content=body)

    # started but tick stale → unhealthy
    if age is not None and age > 30:
        body["ok"] = False
        body["reason"] = f"drain_stale_{int(age)}s"
        return JSONResponse(status_code=503, content=body)

    return body


class TurnRequest(BaseModel):
    conversation_id: str
    message_id: str
    text: str = ""
    has_media: bool = False


def _summarize_result(name: str, result_str: str) -> str:
    """One-line preview of a tool's return value for the trace UI."""
    s = result_str.strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return s[:140]

    if isinstance(obj, dict):
        if "error" in obj:
            return f"error: {obj['error']}"[:140]
        if name == "classify_media":
            return str(obj.get("kind", ""))[:80]
        if name == "extract_passport":
            mrz = obj.get("mrz", {}) or {}
            return f"{mrz.get('name','?')} · {mrz.get('passport','?')}"[:140]
        if name == "extract_voucher":
            ub = obj.get("ub_number") or obj.get("ubNumber") or "?"
            head = obj.get("family_head") or obj.get("familyHead") or "?"
            return f"{ub} · {head}"[:140]
        if name == "match_passport_to_voucher":
            matches = obj.get("matches") or []
            if not matches:
                return "no matches"
            if len(matches) == 1:
                return f"matched {matches[0].get('ub_number','?')}"
            return f"{len(matches)} matches"
        if name == "create_passport":
            if obj.get("conflict") == "cross_agent":
                return "cross-agent collision"
            if obj.get("promoted_from_draft"):
                return "promoted draft → saved"
            if obj.get("duplicate"):
                return "duplicate (already saved)"
            return "saved"
        if name == "upsert_voucher":
            return f"voucher {obj.get('ub_number') or obj.get('id', '?')}"[:140]
        if name == "conversation_state":
            ov = obj.get("open_vouchers") or []
            dp = obj.get("draft_passports") or []
            return f"{len(ov)} open voucher(s) · {len(dp)} draft passport(s)"
        if name == "send_whatsapp":
            return "sent"
        if name == "scan_drafts_for_voucher":
            attached = obj.get("attached") or []
            return f"{len(attached)} draft(s) attached"
        if name == "create_passport_draft":
            return "draft created"
        if name == "escalate_to_l1":
            return f"escalated: {obj.get('reason','')}"[:140]
        first = next(iter(obj.items()), None)
        if first:
            return f"{first[0]}: {str(first[1])[:80]}"
    return s[:140]


def _slice_current_turn(messages) -> list:
    """`result["messages"]` is the whole checkpoint history. Trim to messages
    from the last HumanMessage onward so traces describe only THIS turn's work
    instead of accumulating across every prior turn on the conversation."""
    from langchain_core.messages import HumanMessage
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return list(messages)


def _extract_usage(messages) -> dict:
    """Sum LLM token usage across every AIMessage in the turn. LangChain's
    AIMessage exposes usage_metadata={input_tokens, output_tokens, total_tokens,
    input_token_details: {cache_read, cache_creation}}. Returns zero-dict when
    the model didn't surface usage (e.g. a tool-only turn)."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "ai_messages": 0,
    }
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if not usage:
            continue
        totals["ai_messages"] += 1
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        totals["cache_read_tokens"] += int(details.get("cache_read") or 0)
        totals["cache_creation_tokens"] += int(details.get("cache_creation") or 0)
    return totals


def _extract_tool_calls(messages) -> list[dict]:
    """Pair tool_call requests (on AIMessage) with their ToolMessage results."""
    calls: list[dict] = []
    pending: dict[str, dict] = {}
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            entry = {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args") or {},
                "result_summary": "",
            }
            calls.append(entry)
            pending[entry["id"]] = entry
        tcid = getattr(m, "tool_call_id", None)
        if tcid and tcid in pending:
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(str(p) for p in content)
            pending[tcid]["result_summary"] = _summarize_result(
                pending[tcid]["name"], str(content)
            )
    return calls


def _final_outbound_id(conversation_id: str, started_at_ms: int) -> str | None:
    """Find the OUT message persisted by send_whatsapp during this turn."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            '''SELECT id FROM "Message"
               WHERE "conversationId" = %s AND direction = 'OUT'
                 AND "createdAt" >= to_timestamp(%s / 1000.0)
               ORDER BY "createdAt" DESC LIMIT 1''',
            (conversation_id, started_at_ms),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def _persist_trace(
    conversation_id: str,
    inbound_message_id: str,
    tool_calls: list[dict],
    final_text: str,
    duration_ms: int,
    started_at_ms: int,
) -> None:
    outbound_id = _final_outbound_id(conversation_id, started_at_ms)
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            '''INSERT INTO "AgentTrace"
               (id, "conversationId", "inboundMessageId", "outboundMessageId",
                "toolCalls", "finalText", "durationMs", "startedAt")
               VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, NOW())''',
            (
                str(uuid.uuid4()),
                conversation_id,
                inbound_message_id or None,
                outbound_id,
                json.dumps(tool_calls),
                final_text[:1000] if final_text else None,
                duration_ms,
            ),
        )


@contextlib.contextmanager
def _conversation_lock(conversation_id: str):
    """Postgres advisory lock keyed on conversation_id. Two concurrent /turn
    calls on the same conversation will serialize: the second blocks until the
    first commits its checkpoint. Released on connection close.

    P1 fix: lock_timeout=30s so a wedged worker can't deadlock new drains/turns
    indefinitely. If the lock cannot be acquired in 30s psycopg raises
    LockNotAvailable; caller treats that as a transient skip.
    """
    with _conn() as c, c.cursor() as cur:
        # SET LOCAL only takes effect inside a transaction — psycopg autocommit
        # may be on, so use SET (session-scoped on this connection).
        try:
            cur.execute("SET lock_timeout = '30s'")
        except Exception:
            pass
        cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (conversation_id,))
        try:
            yield
        finally:
            try:
                cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (conversation_id,))
            except Exception:
                pass


def _build_burst_body(conversation_id: str, msgs: list[dict]) -> str:
    lines = [
        "INBOUND BURST",
        f"conversation_id: {conversation_id}",
        f"{len(msgs)} message(s) in this burst:",
    ]
    for i, m in enumerate(msgs, start=1):
        media = m.get("media") or []
        media_count = len(media) if isinstance(media, list) else (1 if m.get("mediaUrl") else 0)
        text = (m.get("body") or "").strip().replace("\n", " ")
        text_preview = text[:80] or "(no text)"
        lines.append(
            f"- [{i}/{len(msgs)}] message_id={m['id']} media={media_count} text={text_preview}"
        )
    return "\n".join(lines)


def _wipe_checkpoint(thread_id: str) -> None:
    """B8: a TimeoutError on `fut.result(...)` does NOT stop the worker thread.
    The zombie may still be in the middle of agent.invoke and will eventually
    write a partial AIMessage (with tool_calls but no matching ToolMessage) to
    PostgresSaver. The next retry then loads malformed state and raises
    "Found AIMessages with tool_calls without ToolMessage".

    Wiping the checkpoint rows for this thread before retry guarantees the
    zombie's eventual write either lands on a clean slate (harmless) or after
    the retry (also fine — it's a different thread). Either way, no malformed
    half-state. We lose conversational memory for the failed turn, which is
    acceptable: we're escalating to L1 anyway.
    """
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
    except Exception as e:
        # Don't let wipe failure mask the original timeout/exception.
        print(f"[checkpoint-wipe] failed for {thread_id}: {e}")
        _capture(e)


def _previous_run_already_replied(conversation_id: str, earliest_inbound_ms: int) -> bool:
    """B7 Option B: idempotency without schema migration.

    `tools.py:_send_counts` is a process-local dict — after a PM2 reload mid-turn,
    the new process loses the guard. If the inbound burst was never marked
    `processedAt` (because the previous process crashed after sending but before
    the UPDATE), the drain loop will re-pick up these inbound rows and re-invoke
    the agent — which will gladly send the same reply a SECOND time.

    Defense: before invoking the agent, check whether any OUT row was already
    persisted on this conversation since the earliest inbound timestamp of this
    burst. If yes, the previous (crashed) run had already replied — skip
    agent.invoke entirely and just stamp processedAt so we don't keep retrying.

    We picked Option B over Option A (turnId + unique constraint) because the
    latter requires a Prisma migration, and prod policy forbids running
    `prisma migrate deploy`.
    """
    # The OUT must post-date BOTH the earliest inbound of the current burst
    # AND the last successfully-processed inbound. Otherwise we mis-flag the
    # legitimate "prior turn replied, then user sent a follow-up" pattern as
    # a crashed retry and silently skip the new turn.
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            '''SELECT 1 FROM "Message"
               WHERE direction = 'OUT'
                 AND "conversationId" = %s
                 AND "createdAt" > to_timestamp(%s / 1000.0)
                 AND "createdAt" > COALESCE(
                   (SELECT MAX("processedAt") FROM "Message"
                    WHERE "conversationId" = %s
                      AND direction = 'IN'
                      AND "processedAt" IS NOT NULL),
                   'epoch'::timestamptz
                 )
               LIMIT 1''',
            (conversation_id, earliest_inbound_ms, conversation_id),
        )
        return cur.fetchone() is not None


def _process_burst(conversation_id: str) -> None:
    """Drain all unprocessed IN messages for this conversation in one turn.

    Bug-fix layout:
      - B6: trace is persisted on EVERY exit path (success/timeout/exception/
        skip), via a single `finally` block.
      - B7: replay-idempotency check before agent.invoke (Option B).
      - B8: checkpoint wipe on TimeoutError before allowing retry.
    """
    started_at_ms = int(time.time() * 1000)
    t0 = time.perf_counter()
    inbound_id_for_trace: str | None = None
    ids: list[str] = []
    result = None  # set on success; None on every error path
    error_meta: dict | None = None  # {error_kind, error_msg, retries}
    skipped_replay = False

    try:
        with _conn() as c, c.cursor() as cur:
            # P1: bound advisory-lock wait so a wedged worker can't deadlock us.
            try:
                cur.execute("SET lock_timeout = '30s'")
            except Exception:
                pass
            try:
                cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (conversation_id,))
            except Exception as e:
                print(f"[drain] lock acquisition failed for {conversation_id}: {e}")
                _capture(e)
                return  # next drain cycle will retry
            try:
                cur.execute(
                    '''SELECT id, body, "mediaUrl", media, "createdAt"
                       FROM "Message"
                       WHERE "conversationId" = %s
                         AND direction = 'IN'
                         AND "processedAt" IS NULL
                       ORDER BY "createdAt" ASC''',
                    (conversation_id,),
                )
                msgs = cur.fetchall()
                if not msgs:
                    return
                ids = [m["id"] for m in msgs]
                inbound_id_for_trace = ids[0]
                earliest_inbound_ms = int(msgs[0]["createdAt"].timestamp() * 1000)

                # B7: idempotency guard. If the prior (crashed) process already
                # sent an OUT for this burst, do NOT re-invoke the agent.
                if _previous_run_already_replied(conversation_id, earliest_inbound_ms):
                    print(f"[drain] skip {conversation_id}: previous run already replied (B7 idempotency)")
                    skipped_replay = True
                    cur.execute(
                        'UPDATE "Message" SET "processedAt" = NOW() WHERE id = ANY(%s::text[])',
                        (ids,),
                    )
                    _clear_failure(conversation_id)
                    error_meta = {
                        "error_kind": "skipped_replay",
                        "error_msg": "previous run already replied; skipped agent.invoke",
                        "retries": 0,
                    }
                    return

                body = _build_burst_body(conversation_id, msgs)
                config = {"configurable": {"thread_id": conversation_id}}

                reset_send_count(conversation_id)  # B6: arm send-once guard
                try:
                    result = _invoke_with_timeout(
                        {"messages": [{"role": "user", "content": body}]},
                        config,
                        TURN_TIMEOUT_S,
                    )
                except concurrent.futures.TimeoutError as e:
                    fails = _bump_failure(conversation_id)
                    print(f"[drain] timeout after {TURN_TIMEOUT_S}s for {conversation_id} (fail #{fails})")
                    _capture(e)
                    # B8: wipe checkpoint so the zombie thread's eventual write
                    # doesn't corrupt the next retry's state.
                    _wipe_checkpoint(conversation_id)
                    error_meta = {
                        "error_kind": "timeout",
                        "error_msg": f"agent.invoke exceeded {TURN_TIMEOUT_S}s",
                        "retries": fails,
                    }
                    if fails >= MAX_TURN_RETRIES:
                        cur.execute(
                            'UPDATE "Message" SET "processedAt" = NOW() WHERE id = ANY(%s::text[])',
                            (ids,),
                        )
                        _clear_failure(conversation_id)
                        _escalate_after_retries(conversation_id, "agent_timeout_retry_exceeded")
                    # else: leave processedAt NULL → next drain cycle retries
                    return
                except Exception as e:
                    fails = _bump_failure(conversation_id)
                    print(f"[drain] agent.invoke failed for {conversation_id} (fail #{fails}): {e}")
                    _capture(e)
                    error_meta = {
                        "error_kind": "exception",
                        "error_msg": str(e)[:300],
                        "retries": fails,
                    }
                    if fails >= MAX_TURN_RETRIES:
                        cur.execute(
                            'UPDATE "Message" SET "processedAt" = NOW() WHERE id = ANY(%s::text[])',
                            (ids,),
                        )
                        _clear_failure(conversation_id)
                        _escalate_after_retries(conversation_id, f"agent_failure_retry_exceeded: {str(e)[:200]}")
                    return

                cur.execute(
                    'UPDATE "Message" SET "processedAt" = NOW() WHERE id = ANY(%s::text[])',
                    (ids,),
                )
                _clear_failure(conversation_id)
            finally:
                try:
                    cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (conversation_id,))
                except Exception:
                    pass
    finally:
        # B6: persist a trace on EVERY exit path. Without this, ops have zero
        # record of timeouts/exceptions — the cases they most need to debug.
        # (Returning from a finally swallows exceptions, but _process_burst is
        # called from a try/except in _drain_loop that already logs/captures —
        # and we explicitly want this path to never bubble.)
        _emit_burst_trace(
            conversation_id=conversation_id,
            inbound_id_for_trace=inbound_id_for_trace,
            t0=t0,
            started_at_ms=started_at_ms,
            result=result,
            error_meta=error_meta,
        )


def _emit_burst_trace(
    conversation_id: str,
    inbound_id_for_trace: str | None,
    t0: float,
    started_at_ms: int,
    result,
    error_meta: dict | None,
) -> None:
    if inbound_id_for_trace is None:
        return  # nothing to trace (no inbound messages found)

    duration_ms = int((time.perf_counter() - t0) * 1000)

    if result is not None:
        # success path — full trace from agent state
        try:
            last = result["messages"][-1]
            final_text = getattr(last, "content", "")
            if isinstance(final_text, list):
                final_text = " ".join(str(p) for p in final_text)
            final_text = str(final_text)
            turn_messages = _slice_current_turn(result["messages"])
            tool_calls = _extract_tool_calls(turn_messages)
            usage = _extract_usage(turn_messages)
            tool_calls.append({
                "id": "_meta", "name": "_meta", "args": {},
                "result_summary": "", "usage": usage,
            })
        except Exception as e:
            print(f"[trace] failed to extract from result: {e}")
            _capture(e)
            final_text = ""
            tool_calls = [{
                "id": "_meta", "name": "_meta", "args": {},
                "result_summary": "",
                "error_kind": "trace_extract_failed",
                "error_msg": str(e)[:300],
            }]
    else:
        # timeout / exception / skipped path — empty reply, error meta only
        final_text = ""
        tool_calls = [{
            "id": "_meta", "name": "_meta", "args": {},
            "result_summary": "",
            **(error_meta or {"error_kind": "unknown", "error_msg": "", "retries": 0}),
        }]

    try:
        _persist_trace(
            conversation_id,
            inbound_id_for_trace or "",
            tool_calls,
            final_text,
            duration_ms,
            started_at_ms,
        )
    except Exception as e:
        print(f"[trace] failed to persist: {e}")
        _capture(e)


DEBOUNCE_S = float(os.environ.get("AGENT_DEBOUNCE_S", "3"))
MEDIA_DEBOUNCE_S = float(os.environ.get("AGENT_MEDIA_DEBOUNCE_S", "15"))
POLL_S = float(os.environ.get("AGENT_POLL_S", "1"))
TURN_TIMEOUT_S = float(os.environ.get("AGENT_TURN_TIMEOUT_S", "60"))
MAX_TURN_RETRIES = int(os.environ.get("AGENT_MAX_TURN_RETRIES", "2"))

# B5 janitor knobs
JANITOR_INTERVAL_S = float(os.environ.get("AGENT_JANITOR_INTERVAL_S", "60"))
JANITOR_STUCK_AFTER_S = int(os.environ.get("AGENT_JANITOR_STUCK_AFTER_S", "300"))  # 5 minutes

# in-memory per-conversation failure counter; resets on success or restart.
# stored on module so the drain loop and /turn share state within one process.
_turn_failures: dict[str, int] = {}
_turn_failures_lock = threading.Lock()


def _bump_failure(cid: str) -> int:
    with _turn_failures_lock:
        n = _turn_failures.get(cid, 0) + 1
        _turn_failures[cid] = n
        return n


def _clear_failure(cid: str) -> None:
    with _turn_failures_lock:
        _turn_failures.pop(cid, None)


# single-thread executor for time-bounded agent.invoke calls
_TURN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("AGENT_TURN_WORKERS", "4")),
    thread_name_prefix="turn",
)


def _invoke_with_timeout(payload: dict, config: dict, timeout_s: float):
    """Run agent.invoke on a worker thread with a hard wall-clock deadline.
    Raises concurrent.futures.TimeoutError if exceeded; the worker keeps running
    in the background but we ignore its eventual result.

    NOTE: see B8 / `_wipe_checkpoint` — the zombie can corrupt PostgresSaver
    state, so callers MUST wipe the checkpoint on TimeoutError before retry.
    """
    fut = _TURN_EXECUTOR.submit(agent.invoke, payload, config)
    return fut.result(timeout=timeout_s)


def _escalate_after_retries(cid: str, reason: str) -> None:
    """Flip bot off and stamp escalationReason. Runs in its own connection
    because the caller's lock-holding connection may already be in a bad state."""
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                '''UPDATE "Conversation"
                   SET "botEnabled" = false,
                       "escalationReason" = %s,
                       "escalatedAt" = NOW()
                   WHERE id = %s''',
                (reason, cid),
            )
    except Exception as e:
        print(f"[escalate] failed to flag {cid}: {e}")
        _capture(e)


def _find_ready_conversations() -> list[str]:
    """Conversations with at least one unprocessed IN message whose latest
    unprocessed arrival is older than the per-burst debounce (= the user has
    been quiet long enough that the burst is likely complete).

    Adaptive: if any unprocessed message in the burst carries media, we wait
    MEDIA_DEBOUNCE_S (default 15s) instead of DEBOUNCE_S (default 3s). WhatsApp
    bursts of passport scans arrive over 20–30 seconds — a 3s window will split
    them into multiple drains and the bot ends up replying twice."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            '''WITH latest AS (
                 SELECT m."conversationId" AS cid,
                        MAX(m."createdAt") AS last_seen,
                        BOOL_OR(
                          m.media IS NOT NULL
                          AND jsonb_typeof(m.media) = 'array'
                          AND jsonb_array_length(m.media) > 0
                        ) AS has_media
                 FROM "Message" m
                 JOIN "Conversation" c ON c.id = m."conversationId"
                 WHERE m.direction = 'IN'
                   AND m."processedAt" IS NULL
                   AND c."botEnabled" = true
                 GROUP BY m."conversationId"
               )
               SELECT cid FROM latest
                WHERE last_seen < NOW() - (
                    CASE WHEN has_media THEN %s ELSE %s END || ' seconds'
                )::interval''',
            (str(MEDIA_DEBOUNCE_S), str(DEBOUNCE_S)),
        )
        return [row["cid"] for row in cur.fetchall()]


def _drain_loop() -> None:
    global _drain_last_tick, _drain_loop_started
    print(f"[drain] loop started (debounce={DEBOUNCE_S}s, poll={POLL_S}s)")
    while True:
        try:
            convs = _find_ready_conversations()
            for cid in convs:
                try:
                    _process_burst(cid)
                except Exception as e:
                    print(f"[drain] burst failed for {cid}: {e}")
                    _capture(e)
        except Exception as e:
            print(f"[drain] poll failed: {e}")
            _capture(e)
        # Heartbeat: /healthz reads this to detect a dead drain thread.
        _drain_last_tick = time.time()
        _drain_loop_started = True
        time.sleep(POLL_S)


def _janitor_loop() -> None:
    """B5: stuck-QUEUED janitor.

    `tools.py:send_whatsapp` writes a Message row with status='QUEUED' BEFORE
    calling Twilio. If the process is killed between the INSERT and the Twilio
    call (PM2 reload, OOM, timeout cancellation), the row stays QUEUED forever:
    no retry, customer never replied to, and ops has no signal.

    This periodic task transitions stale QUEUED OUT rows to FAILED with a
    diagnostic note so L1 can see and manually resend if appropriate. We do NOT
    auto-resend — risk of double-send is too high without human review.
    """
    print(f"[janitor] loop started (every {JANITOR_INTERVAL_S}s, threshold={JANITOR_STUCK_AFTER_S}s)")
    while True:
        try:
            with _conn() as c, c.cursor() as cur:
                # Note: schema has no `twilioError` column on Message — the
                # reason rides in the Sentry message instead. Just transition
                # status; ops sees the row + a corresponding Sentry event.
                cur.execute(
                    '''UPDATE "Message"
                       SET status = 'FAILED'
                       WHERE status = 'QUEUED'
                         AND direction = 'OUT'
                         AND "createdAt" < NOW() - (%s || ' seconds')::interval
                       RETURNING id, "conversationId"''',
                    (str(JANITOR_STUCK_AFTER_S),),
                )
                rows = cur.fetchall()
                if rows:
                    print(f"[janitor] flipped {len(rows)} stuck QUEUED rows to FAILED")
                    for r in rows:
                        _capture_message(
                            f"stuck QUEUED message {r['id']} on conversation {r['conversationId']} "
                            f"flipped to FAILED (agent_crashed_before_send)",
                            level="error",
                        )
        except Exception as e:
            print(f"[janitor] sweep failed: {e}")
            _capture(e)
        time.sleep(JANITOR_INTERVAL_S)


@app.on_event("startup")
def _start_drain() -> None:
    if os.environ.get("AGENT_DRAIN_DISABLED") == "1":
        print("[drain] disabled via AGENT_DRAIN_DISABLED=1")
        return
    t = threading.Thread(target=_drain_loop, name="drain-loop", daemon=True)
    t.start()
    j = threading.Thread(target=_janitor_loop, name="janitor-loop", daemon=True)
    j.start()


@app.post("/turn")
def turn(req: TurnRequest):
    """Manual single-message turn (kept for backwards compat / testing).

    Production path is the drain loop — webhook just records the message and
    the loop coalesces bursts. Calling /turn with a stale message_id is a no-op
    if the drain loop already swept it.
    """
    body = (
        f"INBOUND MESSAGE\n"
        f"conversation_id: {req.conversation_id}\n"
        f"message_id: {req.message_id}\n"
        f"has_media: {req.has_media}\n"
        f"text: {req.text or '(no text body)'}"
    )
    config = {"configurable": {"thread_id": req.conversation_id}}
    started_at_ms = int(time.time() * 1000)
    t0 = time.perf_counter()
    try:
        with _conversation_lock(req.conversation_id):
            reset_send_count(req.conversation_id)  # B6: arm send-once guard
            result = _invoke_with_timeout(
                {"messages": [{"role": "user", "content": body}]},
                config,
                TURN_TIMEOUT_S,
            )
    except concurrent.futures.TimeoutError as e:
        _capture(e)
        # B8: wipe corrupted checkpoint state from the zombie worker.
        _wipe_checkpoint(req.conversation_id)
        raise HTTPException(504, f"agent timed out after {TURN_TIMEOUT_S}s")
    except Exception as e:
        _capture(e)
        raise HTTPException(500, str(e))

    duration_ms = int((time.perf_counter() - t0) * 1000)
    last = result["messages"][-1]
    final_text = getattr(last, "content", "")
    if isinstance(final_text, list):
        final_text = " ".join(str(p) for p in final_text)
    final_text = str(final_text)

    turn_messages = _slice_current_turn(result["messages"])
    tool_calls = _extract_tool_calls(turn_messages)
    usage = _extract_usage(turn_messages)
    tool_calls.append({"id": "_meta", "name": "_meta", "args": {}, "result_summary": "", "usage": usage})

    try:
        _persist_trace(
            req.conversation_id,
            req.message_id,
            tool_calls,
            final_text,
            duration_ms,
            started_at_ms,
        )
    except Exception as e:
        print(f"[trace] failed to persist: {e}")
        _capture(e)

    return {
        "ok": True,
        "final_text": final_text[:500],
        "tool_calls": len(tool_calls),
        "duration_ms": duration_ms,
    }
