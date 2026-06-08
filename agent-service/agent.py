"""Explicit LangGraph StateGraph for Safr-e-Ibadat's WhatsApp inbox.

Replaces the previous create_react_agent implementation with a hand-rolled
pipeline so phase transitions are deterministic Python rather than model
decisions:

    START
      │
      ▼
    classify_node  ── 1 Sonnet call. Decides intent (text-only vs media
      │              burst) and emits a media_plan naming each
      │              (message_id, media_idx, kind) using parallel
      │              classify_media tool calls.
      │
      ▼ Send fan-out: one branch per media item
    extract_one × N  ── No model. Calls extract_passport / extract_voucher /
      │                extract_pdf depending on classified kind. Errors
      │                captured per-item, never raised.
      │
      ▼ join (deterministic_router)
    persist_node    ── No model. Calls finalize_passport_batch (T2.5) which
      │              orchestrates match + blacklist + create + upsert +
      │              scan_drafts in one server-side pass.
      │
      ▼
    reply_node      ── 1 Sonnet call. Composes the WhatsApp reply from the
                       BatchResult, then optionally escalate_to_l1 when
                       requires_l1 is non-empty.

Public interface unchanged: build_agent() returns a Pregel runnable with
agent.invoke({"messages":[...]}, config={"configurable":{"thread_id":cid}})
returning {"messages":[...]}. app.py's _invoke_with_timeout, _extract_tool_calls,
_extract_usage, B6 reset_send_count, and the PostgresSaver checkpointer all
keep working without edits.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import operator
import os
import re
from typing import Annotated, Any, TypedDict
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
import psycopg
import psycopg.rows
from psycopg_pool import ConnectionPool

import tools as tool_module
from tools import (
    ALL_TOOLS,
    classify_media,
    conversation_state,
    create_passport,
    create_passport_draft,
    escalate_to_l1,
    extract_passport,
    extract_pdf,
    extract_voucher,
    lookup_agent,
    lookup_voucher,
    match_passport_to_voucher,
    scan_drafts_for_voucher,
    send_whatsapp,
    upsert_voucher,
)

log = logging.getLogger("agent.graph")


def _psycopg_url(prisma_url: str) -> str:
    """Strip Prisma-only query params so psycopg can use the same DATABASE_URL."""
    p = urlparse(prisma_url)
    keep = [
        (k, v)
        for k, v in parse_qsl(p.query)
        if k not in ("schema", "connection_limit", "pgbouncer", "connect_timeout")
    ]
    return urlunparse(p._replace(query=urlencode(keep)))


DATABASE_URL = _psycopg_url(os.environ["DATABASE_URL"])


# Optional Sentry: capture exceptions when SENTRY_DSN is configured. Same
# pattern as the OCR sidecar + Next.js services. Safe no-op otherwise.
try:
    import sentry_sdk  # type: ignore
    if os.environ.get("SENTRY_DSN") and not getattr(sentry_sdk, "_safr_initialized", False):
        sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=0.0)
        sentry_sdk._safr_initialized = True  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - sentry is best-effort
    sentry_sdk = None  # type: ignore


def _capture(exc: BaseException) -> None:
    """Forward to Sentry when available + configured. Never raises."""
    try:
        if sentry_sdk is not None and os.environ.get("SENTRY_DSN"):
            sentry_sdk.capture_exception(exc)
    except Exception:  # pragma: no cover
        pass


# Module-level read pool reused by ad-hoc query helpers
# (_already_greeted_recently, _lookup_voucher_state). The build_agent() pool
# is owned by PostgresSaver and we don't want to share its lifecycle with
# request-time helpers. Lazy so import-time failures don't crash the worker.
_READ_POOL: ConnectionPool | None = None


def _read_pool() -> ConnectionPool:
    global _READ_POOL
    if _READ_POOL is None:
        _READ_POOL = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": psycopg.rows.dict_row},
        )
    return _READ_POOL


CLASSIFY_SYSTEM_PROMPT = """You are Safr-e-Ibadat's WhatsApp assistant for a Pakistani Umrah B2B operator.
Your job in THIS step is intent classification only. You will not write the user-facing reply here.

You receive an INBOUND BURST listing one or more message_ids (a WhatsApp album fans out into separate webhooks; the drain loop coalesces them into one turn).

Decide the burst's intent and reply with a SINGLE JSON object — no prose, no markdown — with this shape:

{
  "intent": "media" | "text" | "greeting" | "status_query" | "complaint" | "human_request",
  "media_plan": [
    {"message_id": "...", "media_idx": 0, "kind": "passport"|"voucher"|"pdf"|"id_card"|"payment_proof"|"unreadable"|"other"}
  ],
  "text_reply_hint": "<= 200 chars, optional"
}

Rules:
- For every burst entry that has media>0, call classify_media(message_id, media_idx) IN PARALLEL within ONE assistant message — emit M tool_use blocks for M media items. Use the tool's lowercase one-word output verbatim as `kind` in media_plan.
- A message_id with media=0 is text-only and contributes only to intent (never to media_plan).
- If the burst is fully text (no media), set intent based on the text:
    "human_request" if the user asks for a human / complains / sounds upset
    "status_query" if they ask about a UB/voucher/passport status
    "greeting" for chit-chat
    "text" otherwise
- If any media item is a payment_proof, mark intent="media" and include it in media_plan with kind="payment_proof".
- NEVER call send_whatsapp, escalate_to_l1, or any DB-write tool here. The only tools available are: conversation_state, classify_media, lookup_agent, lookup_voucher.
- conversation_state(conversation_id) is allowed at most once if you genuinely need ground-truth; otherwise skip it."""


REPLY_SYSTEM_PROMPT = """You are Safr-e-Ibadat's WhatsApp assistant for a Pakistani Umrah B2B operator.
Your job in THIS step is to write the user-facing reply for the turn that just finished, then (if needed) escalate.

VOICE: natural Pakistani B2B WhatsApp lingo. Friendly, semi-professional, respectful. Sounds like a human teammate at the back office, not a corporate bot.

Hard rules on style:
- One short sentence. Two only if absolutely necessary.
- NO hyphens or em dashes anywhere. Use a comma or period instead. Never write "—" or "-" between phrases.
- No emojis. No markdown. No exclamation marks except the very first greeting on a new conversation.
- No robotic openers like "Hello! How can I help you today?", "I am Safr-e-Ibadat's WhatsApp assistant", "Please share...to get started", "I will get them saved right away".
- Do NOT introduce yourself, do NOT explain what you do, do NOT recite the brand name in replies. The customer already knows who they messaged.
- No progress math, no flag counts, no "saved X of Y", no "held for review", no "please send the remaining". L1 handles flags out of band, only L1 tells the customer to resend.
- No promises about visa timelines.

Language matching (STRICT):
- TURN_DATA.user_lang tells you what the user wrote in: "english", "urdu", "roman_urdu", or "mixed". Reply in the SAME language.
- If user_lang is "english": reply in English ONLY. Do NOT use Roman Urdu words like "ho gaye", "ke against", "save kar diya", "ho gaya hai", "abhi", "ji", "shukria", "thoda", "kr", "kar". Do NOT mix Urdu script. The greeting "Salam" alone is acceptable since it's commonly used in Pakistani English. Everything else after that must be plain English.
- If user_lang is "roman_urdu": reply in Roman Urdu. Example: "Theek hai, UB-100344 ke against save ho gaye."
- If user_lang is "urdu": reply in Urdu script.
- If user_lang is "mixed": match their balance roughly, leaning English for clarity.
- If the user has sent ONLY media (no text yet) so user_lang is "english" by default: reply in English. Do not assume Roman Urdu.

Greeting de-duplication:
- TURN_DATA.already_greeted is true when the bot has already greeted this conversation in a recent turn. If true, do NOT open with another greeting. Just answer or acknowledge directly. A second "hi" from the user does NOT need a fresh greeting.

Multi-voucher handling (use the new aggregate fields, NOT vouchers_processed directly):
- batch_result.vouchers_with_new_saves: vouchers where THIS turn saved at least one fresh passport. Mention these in the ack: "Saved against UB-100344." or "Saved against UB-100344 and UB-100712." Use only their ub_number values.
- batch_result.vouchers_all_duplicates: vouchers where every passport was a duplicate (already on file). Mention as "Already on UB-XXXXXX, nothing new to add."
- batch_result.vouchers_no_passports: vouchers we just upserted but no passports were routed to them yet (likely passports for that UB haven't arrived). "Got UB-XXXXXX, send the passports when ready."
- batch_result.all_duplicates flag: every passport in the whole turn was a duplicate. Use the duplicate phrasing.
- Do NOT mention a voucher with new_saves=0 as "saved against" — that is a hallucination.

Specific-voucher status query:
- TURN_DATA.requested_vouchers (when present) is the list of UB numbers the user explicitly asked about. requested_voucher_states is the parallel list of lookup results.
- For each lookup: if found=true, answer using saved_count, expected_count, missing_count, status, family_head. If found=false and foreign=true, say neutrally "That UB isn't on your account." (do NOT confirm or deny details). If found=false otherwise, say "UB-XXXXXX isn't in our system."
- Answer about THOSE UBs only; do not pivot to a different UB.

Duplicate-burst phrasing:
- When all_duplicates is true, do NOT phrase it as "no new additions this time" or "already saved previously" — that sounds like denial. Use matter-of-fact: "Already on UB-XXXXXX, nothing new to add." Or in Roman Urdu: "Yeh sab pehle se UB-XXXXXX par save hain."

Tone reference (vibe, not literal templates):
- Greetings (first time only, English default): "Salam, please share the voucher or passport whenever ready." / "Hi, how can we help?"
- Confirming a save (use vouchers_with_new_saves): "Got it, saved against UB-100344." / "Saved against UB-100344 and UB-100712." / "Done, all passports saved."
- Voucher upserted but no passports yet: "Got UB-100344, please send the passports when ready."
- Acknowledging without a UB anchor yet: "Got it." / "Received, thanks."
- Single passport, no voucher: "Got NAME, please share the voucher number when ready."
- Cross-voucher ambiguity: "That passport is showing on more than one booking, which one is it for?"
- Entire burst unreadable: "Couldn't read those clearly, kindly resend better photos."
- status_query about a specific UB: answer from requested_voucher_states only.
- human_request / complaint / cross-agent / blacklist / payment_proof: "Noted, our team will take this forward." or "Thanks, our team will handle this from here."

The system has ALREADY done all the database work. You will receive a BATCH_RESULT JSON plus optionally a TEXT_HINT or STATE summary. You must:

1. Call send_whatsapp(conversation_id, text) EXACTLY ONCE, following the voice rules above.

2. If BATCH_RESULT.requires_l1 is non-empty OR intent is human_request/complaint/payment_proof, call escalate_to_l1(conversation_id, reason) AFTER send_whatsapp. The customer-facing text MUST NOT mention this, escalation is internal.

Do NOT call send_whatsapp twice. Do NOT call any other tool. Do NOT include BATCH_RESULT JSON in the reply text. Do NOT reveal internal mechanics."""


class GraphState(TypedDict, total=False):
    """Graph state. `messages` is the only field app.py reads."""
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    sub_agent_id: str | None
    intent: str
    media_plan: list[dict[str, Any]]
    text_reply_hint: str
    state_snapshot: dict[str, Any]
    extract_results: Annotated[list[dict[str, Any]], operator.add]
    batch_result: dict[str, Any]


_BURST_HEADER_RE = re.compile(r"conversation_id:\s*(\S+)")
_BURST_LINE_RE = re.compile(r"\[(\d+)/\d+\]\s+message_id=(\S+)\s+media=(\d+)\s+text=(.*)$")


def _parse_burst_body(body: str) -> dict[str, Any]:
    """Recover the structured burst app.py built in `_build_burst_body`."""
    cid_match = _BURST_HEADER_RE.search(body or "")
    cid = cid_match.group(1) if cid_match else ""
    items: list[dict[str, Any]] = []
    for line in (body or "").splitlines():
        m = _BURST_LINE_RE.search(line)
        if not m:
            continue
        items.append({
            "message_id": m.group(2),
            "media": int(m.group(3)),
            "text": m.group(4).strip(),
        })
    return {"conversation_id": cid, "items": items}


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            return str(content)
    return ""


_UB_RE = re.compile(r"\bUB[\s\-_]*?(\d{3,})\b", re.IGNORECASE)
_ROMAN_URDU_HINTS = (
    "hain", "kar", "nahi", "nahin", "kya", "hai ", "hain.", "kr ", "krna",
    "shukria", "shukriya", "bhai", "kab ", "kahan", "kaise", "abhi",
    "kitne", "kitna", "thoda", "thik", "theek", "ji ", "jee ", "ke ",
    "ka ", "ki ", "hum ", "tum ", "aap ", "mera ", "meri ", "mujhe",
)
_URDU_SCRIPT_RE = re.compile(r"[؀-ۿ]")


def _detect_user_lang(text: str) -> str:
    """Quick heuristic: english | urdu | roman_urdu | mixed."""
    if not text:
        return "english"
    s = text.lower()
    has_urdu_script = bool(_URDU_SCRIPT_RE.search(text))
    has_roman_urdu = any(w in s for w in _ROMAN_URDU_HINTS)
    has_ascii_words = bool(re.search(r"[a-z]{3,}", s))
    if has_urdu_script and has_ascii_words:
        return "mixed"
    if has_urdu_script:
        return "urdu"
    if has_roman_urdu:
        return "roman_urdu"
    return "english"


def _lang_instruction(user_lang: str) -> str:
    """Per-turn deterministic language directive injected directly into the
    user-side instruction. The system prompt has language rules too, but
    Sonnet 4.6 has a strong prior to use Roman Urdu when it sees Pakistani
    business context, so we double-up with a turn-specific reminder."""
    if user_lang == "english":
        return (
            "WRITE THE REPLY IN ENGLISH. The user has written in English (or "
            "sent only media), so reply in plain English. Do NOT use any Roman "
            "Urdu words, including but not limited to: 'ho gaye', 'ho gaya', "
            "'ho gayi', 'ke against', 'ka', 'ki', 'ke', 'aur', 'bhi', 'jab', "
            "'hon', 'bhej dein', 'kar dein', 'kar diya', 'mil gaya', 'kya', "
            "'hai', 'hain', 'shukria', 'thoda', 'abhi', 'jee', 'ji', 'sir ji'. "
            "The greeting 'Salam' on a brand-new conversation only is acceptable; "
            "everything else stays English."
        )
    if user_lang == "roman_urdu":
        return (
            "WRITE THE REPLY IN ROMAN URDU. The user wrote in Roman Urdu, so "
            "match them. Example phrasing: 'Theek hai, UB-XXXXXX ke against "
            "save ho gaye hain.'"
        )
    if user_lang == "urdu":
        return "WRITE THE REPLY IN URDU SCRIPT. The user wrote in Urdu, so match them."
    if user_lang == "mixed":
        return (
            "WRITE THE REPLY IN A LIGHT MIX of English and Roman Urdu, leaning "
            "English for clarity. The user mixed languages."
        )
    return "WRITE THE REPLY IN ENGLISH."


def _extract_requested_ubs(text: str) -> list[str]:
    """Pull every UB number out of the user's free-text question, deduped and
    in order of first appearance. Accepts 'UB-180139', 'ub 180139', 'ub180139'.

    Guards against false positives from the burst body itself: the burst
    contains structured lines like `[1/3] message_id=... media=... text=...`
    plus a `conversation_id:` header. We strip those before matching so a
    voucher number bleeding into a `text=` preview (rare today, but possible
    once previews include richer content) cannot be mis-read as a user
    question about that UB. Only lines that look like real user-typed text
    are scanned."""
    if not text:
        return []
    candidates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip burst structural lines: `[N/M] message_id=... text=...` and the
        # `conversation_id: ...` header. Anything else is treated as user text.
        if stripped.startswith("[") or "message_id=" in stripped:
            continue
        if stripped.lower().startswith("conversation_id:"):
            continue
        candidates.append(stripped)
    if not candidates:
        # Fallback: caller passed plain user text without burst framing.
        candidates = [text]
    seen: list[str] = []
    for line in candidates:
        for m in _UB_RE.finditer(line):
            ub = f"UB-{m.group(1)}"
            if ub not in seen:
                seen.append(ub)
    return seen


_GREETING_START_RE = re.compile(r"^\s*(salam|hi|hello|hey|assalam)[\s,!.،]", re.IGNORECASE)


def _already_greeted_recently(conversation_id: str, lookback: int = 5) -> bool:
    """Did the bot send a greeting-style reply in the last `lookback` outbound
    messages on this conversation? Used to suppress duplicate 'Salam, kindly
    share...' responses when the user pings 'hi' multiple times.

    Anchors greeting words at the START of the body so customer names like
    'Salam Khan' inside a save-confirmation don't trigger a false positive.
    The 'how can we help' phrasing stays a substring check (less ambiguous)."""
    if not conversation_id:
        return False
    try:
        with _read_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''SELECT body FROM "Message"
                       WHERE "conversationId" = %s AND direction = 'OUT'
                       ORDER BY "createdAt" DESC LIMIT %s''',
                    (conversation_id, lookback),
                )
                rows = cur.fetchall()
    except Exception as exc:
        log.warning("greeting-check query failed: %s", exc)
        _capture(exc)
        return False
    for r in rows:
        body = (r.get("body") or "")
        if _GREETING_START_RE.match(body):
            return True
        low = body.lower()
        if "how can we help" in low or "how can i help" in low:
            return True
    return False


def _lookup_voucher_state(ub_number: str, sub_agent_id: str | None) -> dict[str, Any]:
    """Fetch a single voucher's snapshot for the reply node when the user asks
    about a specific UB.

    Privacy: scoped to the conversation's sub-agent. If the UB belongs to a
    different sub-agent we return {found: False, foreign: True} — never reveal
    family head or counts cross-agent. If sub_agent_id is None (e.g. an
    unmapped conversation) we still scope strictly: only return vouchers with
    NULL subAgentId, treating the rest as foreign."""
    if not ub_number:
        return {"found": False, "ub_number": ub_number}
    try:
        with _read_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''SELECT v.id, v."ubNumber", v.status, v."familyHead",
                              v."subAgentId",
                              v."expectedMutamers",
                              COUNT(p.id) FILTER (WHERE p.id IS NOT NULL) AS saved_count
                       FROM "Voucher" v
                       LEFT JOIN "Passport" p ON p."voucherId" = v.id
                       WHERE v."ubNumber" = %s
                       GROUP BY v.id''',
                    (ub_number,),
                )
                row = cur.fetchone()
    except Exception as exc:
        log.warning("voucher lookup failed for %s: %s", ub_number, exc)
        _capture(exc)
        return {"found": False, "ub_number": ub_number, "error": str(exc)}
    if not row:
        return {"found": False, "ub_number": ub_number}
    # cross-agent privacy gate
    if (row.get("subAgentId") or None) != (sub_agent_id or None):
        return {"found": False, "foreign": True, "ub_number": ub_number}
    expected = row.get("expectedMutamers") or []
    expected_count = len(expected) if isinstance(expected, list) else 0
    saved = int(row.get("saved_count") or 0)
    expected_pnos = {
        (m.get("passport") or "").strip().upper()
        for m in expected if isinstance(m, dict) and m.get("passport")
    }
    saved_pnos: set[str] = set()
    try:
        with _read_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT "passportNumber" FROM "Passport" WHERE "voucherId" = %s',
                    (row["id"],),
                )
                saved_pnos = {
                    (r.get("passportNumber") or "").strip().upper()
                    for r in cur.fetchall()
                }
    except Exception as exc:
        log.warning("voucher passport lookup failed for %s: %s", ub_number, exc)
        _capture(exc)
    missing = sorted(expected_pnos - saved_pnos) if expected_pnos else []
    # find missing names too, when available, for the reply
    missing_with_names: list[dict[str, str]] = []
    if missing:
        for m in expected:
            if not isinstance(m, dict):
                continue
            pno = (m.get("passport") or "").strip().upper()
            if pno in missing:
                missing_with_names.append({
                    "passport_number": pno,
                    "name": (m.get("name") or "").strip(),
                })
    return {
        "found": True,
        "ub_number": row["ubNumber"],
        "status": row["status"],
        "family_head": row.get("familyHead"),
        "saved_count": saved,
        "expected_count": expected_count,
        "missing_count": max(0, expected_count - saved),
        "missing": missing_with_names,
    }


def _conversation_id_from_state(state: GraphState) -> str:
    cid = state.get("conversation_id") or ""
    if cid:
        return cid
    parsed = _parse_burst_body(_last_human_text(state.get("messages") or []))
    return parsed.get("conversation_id") or ""


CLASSIFY_TOOLS = [conversation_state, classify_media, lookup_agent, lookup_voucher]
REPLY_TOOLS = [send_whatsapp, escalate_to_l1]


def _make_model() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=1024)


def _cached_system(prompt: str) -> SystemMessage:
    """Anthropic ephemeral cache_control on the system block."""
    return SystemMessage(
        content=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
    )


_TOOL_BY_NAME: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


def _run_tool_call(tc: dict[str, Any]) -> ToolMessage:
    name = tc.get("name") or ""
    args = tc.get("args") or {}
    tool_call_id = tc.get("id") or ""
    impl = _TOOL_BY_NAME.get(name)
    if impl is None:
        return ToolMessage(
            content=json.dumps({"error": f"unknown tool: {name}"}),
            tool_call_id=tool_call_id, name=name,
        )
    try:
        out = impl.invoke(args)
    except Exception as exc:
        log.exception("tool %s failed", name)
        _capture(exc)
        out = {"error": f"tool {name} raised: {exc}"}
    if not isinstance(out, str):
        try:
            out_str = json.dumps(out, default=str)
        except (TypeError, ValueError):
            out_str = str(out)
    else:
        out_str = out
    return ToolMessage(content=out_str, tool_call_id=tool_call_id, name=name)


def _execute_tool_calls(ai_msg: AIMessage) -> list[ToolMessage]:
    return [_run_tool_call(tc) for tc in (ai_msg.tool_calls or [])]


def _classify_node(state: GraphState) -> dict[str, Any]:
    """Pure Python: classify all media in parallel, derive intent heuristically.
    No model call. The reply_node still has the LLM for prose composition.

    For media bursts we know `kind` from classify_media (a deterministic vision
    classifier), and intent is mechanically derivable from the kinds + any text
    items. The model's previous role here ("decide intent + emit media_plan")
    was redundant given that classify_media already returns the kind."""
    messages = list(state.get("messages") or [])
    last_text = _last_human_text(messages)
    parsed = _parse_burst_body(last_text)
    cid = parsed.get("conversation_id") or state.get("conversation_id") or ""
    items = parsed.get("items") or []

    # text-only short-circuit: no vision, no model.
    if items and all(int(it.get("media") or 0) == 0 for it in items):
        return {
            "conversation_id": cid,
            "intent": _heuristic_text_intent(items),
            "media_plan": [],
            "text_reply_hint": " | ".join(
                (it.get("text") or "").strip() for it in items if it.get("text")
            )[:400],
            "messages": [],
        }

    # Media path: classify every media item in parallel via threadpool.
    media_items = [
        {"message_id": it["message_id"], "media_idx": 0}
        for it in items if int(it.get("media") or 0) > 0
    ]
    plan: list[dict[str, Any]] = []
    if media_items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(media_items))) as ex:
            futures = {
                ex.submit(classify_media.invoke, mi): mi for mi in media_items
            }
            for fut in concurrent.futures.as_completed(futures):
                mi = futures[fut]
                try:
                    kind = fut.result()
                    if not isinstance(kind, str):
                        kind = "other"
                except Exception as exc:
                    log.warning("classify_media failed for %s: %s", mi.get("message_id"), exc)
                    _capture(exc)
                    kind = "passport"  # safest fallback (will OCR; bad images route to recover)
                plan.append({
                    "message_id": mi["message_id"],
                    "media_idx": mi["media_idx"],
                    "kind": (kind or "").strip().lower(),
                })

    # Preserve burst order (futures complete out of order).
    order = {m["message_id"]: i for i, m in enumerate(media_items)}
    plan.sort(key=lambda p: order.get(p["message_id"], 0))

    intent = _intent_from_plan_and_text(plan, items)
    text_hint = " | ".join(
        (it.get("text") or "").strip() for it in items if it.get("text")
    )[:400]

    # Synthesize an AIMessage so app.py's _extract_tool_calls still surfaces
    # the classify_media calls in the trace (matching the pre-T2.6 shape).
    synthetic_calls = []
    synthetic_results = []
    for p in plan:
        tcid = f"call_{abs(hash((cid, p['message_id'], p['media_idx']))) & 0xFFFFFFFF:x}"
        synthetic_calls.append({
            "id": tcid,
            "name": "classify_media",
            "args": {"message_id": p["message_id"], "media_idx": p["media_idx"]},
        })
        synthetic_results.append(ToolMessage(
            content=p["kind"], tool_call_id=tcid, name="classify_media",
        ))
    appended: list[AnyMessage] = []
    if synthetic_calls:
        appended.append(AIMessage(content="", tool_calls=synthetic_calls))
        appended.extend(synthetic_results)

    return {
        "conversation_id": cid,
        "intent": intent,
        "media_plan": plan,
        "text_reply_hint": text_hint,
        "messages": appended,
    }


def _intent_from_plan_and_text(plan: list[dict[str, Any]], items: list[dict[str, Any]]) -> str:
    """Mechanical intent derivation from classify results + any text items."""
    kinds = {(p.get("kind") or "").lower() for p in plan}
    if "payment_proof" in kinds:
        return "media"  # routes through persist; reply_node escalates from requires_l1
    if kinds & {"passport", "voucher", "pdf"}:
        return "media"
    text_intent = _heuristic_text_intent(items)
    if text_intent != "text":
        return text_intent
    # All media classified as id_card/other/unreadable — treat as text intent
    # so reply_node asks for the right document.
    return "media" if kinds else "text"


def _heuristic_text_intent(items: list[dict[str, Any]]) -> str:
    text = " ".join((it.get("text") or "").lower() for it in items)
    if any(w in text for w in ("human", "agent", "complaint", "angry", "speak to", "talk to")):
        return "human_request"
    if any(w in text for w in ("ub-", "ub ", "voucher", "status", "passport")):
        return "status_query"
    if any(w in text for w in ("hi", "hello", "salam", "assalam", "thanks")):
        return "greeting"
    return "text"


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_plan_from_ai(
    ai: AIMessage | None, burst_items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, str]:
    text = ""
    if ai is not None:
        c = ai.content
        if isinstance(c, list):
            text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        else:
            text = str(c or "")

    parsed: dict[str, Any] | None = None
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except (TypeError, ValueError):
            parsed = None

    if parsed and isinstance(parsed, dict):
        plan = [p for p in (parsed.get("media_plan") or []) if isinstance(p, dict) and p.get("message_id")]
        intent = parsed.get("intent") or ("media" if plan else "text")
        hint = (parsed.get("text_reply_hint") or "")[:400]
        return plan, str(intent), str(hint)

    # heuristic fallback: assume passport (most common) for media items.
    plan = [
        {"message_id": it["message_id"], "media_idx": 0, "kind": "passport"}
        for it in burst_items if int(it.get("media") or 0) > 0
    ]
    intent = "media" if plan else _heuristic_text_intent(burst_items)
    return plan, intent, ""


def _route_to_extract(state: GraphState) -> Any:
    """Conditional edge: emit one Send per extractable media item, OR jump
    to persist when nothing is extractable. Empty Send list deadlocks
    Pregel — returning the literal node name routes directly."""
    plan = state.get("media_plan") or []
    extractable = [p for p in plan if (p.get("kind") or "").lower() in ("passport", "voucher", "pdf")]
    if not extractable:
        return "persist"
    return [
        Send("extract_one", {"plan": p, "conversation_id": state.get("conversation_id", "")})
        for p in extractable
    ]


def _extract_one_node(packet: dict[str, Any]) -> dict[str, Any]:
    """Pure Python, runs in parallel via Send. Errors are captured per-item."""
    plan = packet.get("plan") or {}
    msg_id = plan.get("message_id") or ""
    idx = int(plan.get("media_idx") or 0)
    kind = (plan.get("kind") or "").lower()

    result: dict[str, Any] = {
        "message_id": msg_id, "media_idx": idx, "kind": kind,
        "ok": False, "data": None, "error": None,
    }
    try:
        if kind == "passport":
            result["data"] = extract_passport.invoke({"message_id": msg_id, "media_idx": idx})
        elif kind == "voucher":
            result["data"] = extract_voucher.invoke({"message_id": msg_id, "media_idx": idx})
        elif kind == "pdf":
            result["data"] = extract_pdf.invoke({"message_id": msg_id, "media_idx": idx})
        else:
            result["error"] = f"unsupported kind: {kind}"
            return {"extract_results": [result]}
        if isinstance(result["data"], dict) and result["data"].get("error"):
            result["error"] = str(result["data"].get("error"))
        else:
            result["ok"] = True
    except Exception as exc:
        log.exception("extract_one failed for %s", msg_id)
        _capture(exc)
        result["error"] = str(exc)
    return {"extract_results": [result]}


def _persist_node(state: GraphState) -> dict[str, Any]:
    """No model. Walks extract_results, dispatches finalize_passport_batch.
    Builds the passport list in the FLAT shape T2.5 expects (passport_number,
    surname, given_names, ...) and the voucher dict with mapped keys
    (mutamers → expected_mutamers)."""
    cid = _conversation_id_from_state(state)
    extracts = list(state.get("extract_results") or [])
    intent = state.get("intent") or ""

    try:
        snapshot = conversation_state.invoke({"conversation_id": cid}) if cid else {}
    except Exception as exc:
        log.warning("conversation_state failed in persist_node: %s", exc)
        _capture(exc)
        snapshot = {"error": str(exc)}

    sub_agent = (snapshot or {}).get("sub_agent") or {}
    sub_agent_id = sub_agent.get("id") if isinstance(sub_agent, dict) else None

    # Walk extract results into FLAT passport dicts and FULL voucher dicts.
    voucher_data: list[dict[str, Any]] = []
    passports: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []

    def _flatten_passport(source_msg_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        primary = (data.get("primary") or {}) if isinstance(data, dict) else {}
        mrz_check = (data.get("mrz_check") or {}) if isinstance(data, dict) else {}
        verified = bool(data.get("verified")) if isinstance(data, dict) else False
        pno = (primary.get("passport_number") or "").strip().upper()
        if not pno and not (primary.get("surname") or primary.get("given_names")):
            return None
        return {
            "passport_number": pno,
            "surname": primary.get("surname") or "",
            "given_names": primary.get("given_names") or "",
            "date_of_birth": primary.get("date_of_birth") or "",
            "sex": primary.get("gender") or primary.get("sex") or "",
            "expiry_date": primary.get("date_of_expiry") or primary.get("expiry_date") or "",
            "cnic": primary.get("cnic"),
            "place_of_birth": primary.get("place_of_birth"),
            "booklet_number": primary.get("booklet_number"),
            "source_message_id": source_msg_id,
            "verified": verified,
            "verification_notes": mrz_check.get("reason"),
        }

    for r in extracts:
        kind = r.get("kind") or ""
        if not r.get("ok"):
            bad.append(r)
            continue
        data = r.get("data") or {}
        msg_id = r.get("message_id") or ""
        if kind == "voucher":
            voucher_data.append({"source_message_id": msg_id, **data})
        elif kind == "passport":
            flat = _flatten_passport(msg_id, data)
            if flat:
                passports.append(flat)
        elif kind == "pdf":
            # PDF: split each page into its own bucket.
            for page in data.get("pages") or []:
                page_kind = (page.get("kind") or "").lower()
                if page_kind == "passport":
                    primary = page.get("primary") or {}
                    if not primary and (page.get("mrz") or page.get("viz")):
                        primary = {**(page.get("viz") or {}), **(page.get("mrz") or {})}
                    if not primary.get("passport_number"):
                        continue
                    flat = _flatten_passport(msg_id, {
                        "primary": primary,
                        "mrz_check": page.get("mrz_check") or {},
                        "verified": bool(page.get("verified")),
                    })
                    if flat:
                        passports.append(flat)
                elif page_kind == "voucher":
                    # extract_pdf only classifies pages — it doesn't parse the
                    # voucher's structured fields. Follow up with extract_voucher
                    # (vision-first path) on the same Message so we get
                    # ub_number, family_head, expected_mutamers etc.
                    if page.get("ub_number") or page.get("mutamers"):
                        voucher_data.append({"source_message_id": msg_id, **page})
                    else:
                        try:
                            ev = extract_voucher.invoke({"message_id": msg_id})
                            if isinstance(ev, dict) and not ev.get("error") and ev.get("ub_number"):
                                voucher_data.append({"source_message_id": msg_id, **ev})
                        except Exception as exc:
                            log.warning("extract_voucher fallback failed for pdf %s: %s", msg_id, exc)
                            _capture(exc)

    # Build the vouchers list in finalize_passport_batch's expected shape
    # (mutamers → expected_mutamers, drop None keys, drop entries without a
    # readable ub_number — OCR sometimes produces partial voucher pages).
    voucher_args: list[dict[str, Any]] = []
    for v in voucher_data:
        # Carry through OCR provenance: extract_voucher tags _scan_url + _source_kind
        # so upsert_voucher can persist scanUrls + sourceKind for the unified
        # review surface. Fall back to deriving from source_message_id when the
        # extract_pdf path skipped the tags.
        scan_url = v.get("_scan_url")
        if not scan_url and v.get("source_message_id"):
            scan_url = f"/api/media/{v['source_message_id']}/0"
        cleaned = {
            "ub_number": v.get("ub_number"),
            "family_head": v.get("family_head"),
            "agency_name": v.get("agency_name"),
            "expected_mutamers": v.get("expected_mutamers") or v.get("mutamers"),
            "voucher_date": v.get("voucher_date"),
            "package_nights": v.get("package_nights"),
            "airline": v.get("airline"),
            "outbound_flight": v.get("outbound_flight"),
            "outbound_date": v.get("outbound_date"),
            "return_date": v.get("return_date"),
            "departure_city": v.get("departure_city"),
            "arrival_city": v.get("arrival_city"),
            "hotel_name": v.get("hotel_name"),
            "hotel_city": v.get("hotel_city"),
            "checkin_date": v.get("checkin_date"),
            "checkout_date": v.get("checkout_date"),
            "room_type": v.get("room_type"),
            "room_count": v.get("room_count"),
            "_scan_url": scan_url,
            "_source_kind": v.get("_source_kind"),
        }
        cleaned = {k: val for k, val in cleaned.items() if val is not None}
        if cleaned.get("ub_number"):
            voucher_args.append(cleaned)

    finalize = getattr(tool_module, "finalize_passport_batch", None)
    if finalize is not None and (passports or voucher_args):
        finalize_args: dict[str, Any] = {
            "passports": passports,
            "sub_agent_id": sub_agent_id,
            "conversation_id": cid,
            "vouchers": voucher_args,
        }
        try:
            batch = finalize.invoke(finalize_args)
        except Exception as exc:
            log.exception("finalize_passport_batch failed")
            _capture(exc)
            batch = {"error": str(exc), "outcomes": [], "requires_l1": []}
    else:
        batch = _finalize_via_primitives(
            passports=passports, vouchers=voucher_data,
            sub_agent_id=sub_agent_id, conversation_id=cid,
        )

    batch.setdefault("outcomes", [])
    batch.setdefault("requires_l1", [])
    batch["unreadable_count"] = len(bad)
    batch["intent"] = intent
    batch["bad"] = [
        {"message_id": b.get("message_id"), "kind": b.get("kind"), "error": b.get("error")}
        for b in bad
    ]

    return {
        "state_snapshot": snapshot,
        "sub_agent_id": sub_agent_id,
        "batch_result": batch,
        "messages": [],
    }


def _finalize_via_primitives(
    passports: list[dict[str, Any]],
    vouchers: list[dict[str, Any]],
    sub_agent_id: str | None,
    conversation_id: str,
) -> dict[str, Any]:
    """Fallback when finalize_passport_batch (T2.5) hasn't shipped yet.
    Mirrors the {voucher_id, drafts_attached, outcomes, requires_l1} shape."""
    voucher_id: str | None = None
    drafts_attached: list[dict[str, Any]] = []

    for v in vouchers:
        try:
            uv = upsert_voucher.invoke({
                "ub_number": v.get("ub_number") or "",
                "family_head": v.get("family_head"),
                "agency_name": v.get("agency_name"),
                "expected_mutamers": v.get("expected_mutamers") or v.get("mutamers"),
                "sub_agent_id": sub_agent_id,
                "scan_url": v.get("_scan_url"),
                "source_kind": v.get("_source_kind"),
                "airline": v.get("airline"),
                "outbound_flight": v.get("outbound_flight"),
                "outbound_date": v.get("outbound_date"),
                "return_date": v.get("return_date"),
                "departure_city": v.get("departure_city"),
                "arrival_city": v.get("arrival_city"),
                "hotel_name": v.get("hotel_name"),
                "hotel_city": v.get("hotel_city"),
                "checkin_date": v.get("checkin_date"),
                "checkout_date": v.get("checkout_date"),
                "room_type": v.get("room_type"),
                "room_count": v.get("room_count"),
            })
            if isinstance(uv, dict) and uv.get("error") == "cross_agent_conflict":
                log.warning(
                    "upsert_voucher cross-agent block: %s belongs to %s",
                    uv.get("ub_number"), uv.get("existing_agent"),
                )
                continue
            vid = (uv or {}).get("id")
            if vid and not voucher_id:
                voucher_id = vid
            if vid:
                try:
                    sd = scan_drafts_for_voucher.invoke({"voucher_id": vid})
                    drafts_attached.extend((sd or {}).get("attached") or [])
                except Exception as exc:
                    log.warning("scan_drafts failed: %s", exc)
                    _capture(exc)
        except Exception as exc:
            log.exception("upsert_voucher failed for %s", v.get("ub_number"))
            _capture(exc)

    outcomes: list[dict[str, Any]] = []
    requires_l1: list[str] = []

    for p in passports:
        pn = p.get("passport_number") or ""
        if not pn:
            continue
        try:
            m = match_passport_to_voucher.invoke(
                {"passport_number": pn, "sub_agent_id": sub_agent_id}
            )
        except Exception as exc:
            outcomes.append({"passport_number": pn, "status": "error", "error": str(exc)})
            continue

        match = (m or {}).get("match")
        matches = (m or {}).get("matches") or []

        try:
            if match:
                cp = create_passport.invoke({**p, "voucher_id": match["id"]})
                if (cp or {}).get("conflict") == "cross_agent":
                    requires_l1.append(pn)
                    outcomes.append({
                        "passport_number": pn, "status": "cross_agent_conflict",
                        "voucher_id": None, "voucher_ub": None,
                        "passport_id": None, "flagged": True,
                        "detail": cp.get("existing"), "error": None,
                    })
                else:
                    outcomes.append({
                        "passport_number": pn,
                        "status": "duplicate" if cp.get("duplicate") else (
                            "promoted_from_draft" if cp.get("promoted_from_draft") else "saved"
                        ),
                        "voucher_id": match.get("id"),
                        "voucher_ub": match.get("ubNumber") or match.get("ub_number"),
                        "passport_id": cp.get("id"),
                        "flagged": cp.get("screening_status") == "FLAGGED",
                        "detail": None, "error": None,
                    })
            elif matches:
                outcomes.append({
                    "passport_number": pn, "status": "multi_match",
                    "voucher_id": None, "voucher_ub": None,
                    "passport_id": None, "flagged": False,
                    "detail": {"matches": matches}, "error": None,
                })
            else:
                cd = create_passport_draft.invoke(p)
                outcomes.append({
                    "passport_number": pn, "status": "saved_as_draft",
                    "voucher_id": None, "voucher_ub": None,
                    "passport_id": (cd or {}).get("id"),
                    "flagged": (cd or {}).get("screening_status") == "FLAGGED",
                    "detail": None, "error": (cd or {}).get("error"),
                })
        except Exception as exc:
            log.exception("create_passport(_draft) failed for %s", pn)
            _capture(exc)
            outcomes.append({"passport_number": pn, "status": "error", "error": str(exc)})

    # Adapt the legacy fallback output to the new finalize_passport_batch
    # shape so reply_node + _trim_for_prompt see consistent keys.
    voucher_ids = [voucher_id] if voucher_id else []
    vouchers_processed: list[dict[str, Any]] = []
    if voucher_id:
        ub = next(
            (o.get("voucher_ub") for o in outcomes if o.get("voucher_id") == voucher_id),
            None,
        )
        passport_count = sum(1 for o in outcomes if o.get("voucher_id") == voucher_id)
        vouchers_processed.append({
            "voucher_id": voucher_id,
            "ub_number": ub,
            "passport_count": passport_count,
        })
    return {
        "voucher_ids": voucher_ids,
        "vouchers_processed": vouchers_processed,
        "drafts_attached": drafts_attached,
        "outcomes": outcomes,
        "requires_l1": requires_l1,
    }


def _post_persist_router(state: GraphState) -> str:
    """Recovery only triggers when the entire burst was unreadable AND
    nothing got persisted. Text-only intents go straight to reply.

    Decision on multi_match-only bursts: if `outcomes` is non-empty but every
    outcome is `multi_match` (cross-voucher ambiguity, no save, no voucher_id
    yet), we still send to `reply`, NOT `recover`. The image WAS readable,
    so the apology copy ('Couldn't read those clearly, please resend') would
    be wrong — instead reply_node uses the multi_match phrasing ('That
    passport is showing on more than one booking, which one is it for?').
    The current condition `not (outcomes or voucher_ids)` already preserves
    this, since multi_match outcomes are present in `outcomes`. Only the
    truly-zero-signal case (all bad, nothing persisted, no outcomes at all)
    falls through to recover."""
    batch = state.get("batch_result") or {}
    bad = batch.get("bad") or []
    intent = batch.get("intent") or state.get("intent") or ""
    if intent in ("text", "greeting", "status_query", "human_request", "complaint"):
        return "reply"
    if bad and not (batch.get("outcomes") or batch.get("voucher_ids")):
        return "recover"
    return "reply"


def _recover_node(state: GraphState) -> dict[str, Any]:
    """All-unreadable burst: fixed apology, no model."""
    cid = _conversation_id_from_state(state)
    bad = (state.get("batch_result") or {}).get("bad") or []
    if len(bad) == 1:
        text = "Couldn't read the image you sent — please send a clearer photo."
    else:
        text = f"Couldn't read {len(bad)} of the images you sent — please resend clearer photos."
    return _send_only(cid, text)


def _reply_node(state: GraphState) -> dict[str, Any]:
    """One Sonnet call: composes the WhatsApp reply from the BatchResult."""
    cid = _conversation_id_from_state(state)
    batch = state.get("batch_result") or {}
    snapshot = state.get("state_snapshot") or {}
    intent = state.get("intent") or batch.get("intent") or "text"
    hint = state.get("text_reply_hint") or ""

    messages = list(state.get("messages") or [])
    last_text = _last_human_text(messages)
    sub_agent = (snapshot.get("sub_agent") or {}) if isinstance(snapshot, dict) else {}
    sub_agent_id = sub_agent.get("id") if isinstance(sub_agent, dict) else None
    payload = {
        "intent": intent,
        "text_hint": hint,
        "batch_result": _trim_for_prompt(batch),
        "state": _trim_state(snapshot),
        "conversation_id": cid,
        "user_lang": _detect_user_lang(last_text),
        "already_greeted": _already_greeted_recently(cid),
    }
    requested_ubs = _extract_requested_ubs(last_text)
    if requested_ubs:
        payload["requested_vouchers"] = requested_ubs
        payload["requested_voucher_states"] = [
            _lookup_voucher_state(ub, sub_agent_id) for ub in requested_ubs
        ]
    lang_directive = _lang_instruction(payload["user_lang"])
    instruction = (
        f"{lang_directive}\n\n"
        "Compose the WhatsApp reply for this turn now. Call send_whatsapp exactly once, "
        "then escalate_to_l1 if requires_l1 is non-empty or the intent demands it.\n\n"
        f"TURN_DATA:\n{json.dumps(payload, default=str)[:6000]}\n\n"
        f"REMINDER: {lang_directive}"
    )

    model = _make_model().bind_tools(REPLY_TOOLS)
    convo: list[AnyMessage] = [
        _cached_system(REPLY_SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]
    appended: list[AnyMessage] = []
    sent_once = False

    # Up to 3 model turns: send → [tool result] → escalate → [tool result] → final.
    for _ in range(3):
        ai = model.invoke(convo)
        appended.append(ai)
        convo.append(ai)
        if not ai.tool_calls:
            break
        tool_msgs: list[ToolMessage] = []
        for tc in ai.tool_calls:
            if tc.get("name") == "send_whatsapp":
                if sent_once:
                    tool_msgs.append(ToolMessage(
                        content=json.dumps({
                            "error": "already_sent_this_turn",
                            "hint": "do not call send_whatsapp again",
                        }),
                        tool_call_id=tc.get("id") or "",
                        name="send_whatsapp",
                    ))
                    continue
                sent_once = True
            tool_msgs.append(_run_tool_call(tc))
        appended.extend(tool_msgs)
        convo.extend(tool_msgs)

    if not sent_once:
        # Language-aware fallback. Brand-new conversations (no prior messages)
        # default to English. The em dash that previously lived here violated
        # the system prompt's "NO hyphens or em dashes" rule.
        fallback_lang = (
            _detect_user_lang(last_text) if (messages and last_text) else "english"
        )
        if fallback_lang == "roman_urdu":
            fallback_text = "Shukria, message mil gaya."
        elif fallback_lang == "urdu":
            fallback_text = "شکریہ، آپ کا پیغام مل گیا۔"
        elif fallback_lang == "mixed":
            fallback_text = "Thanks, message mil gaya."
        else:
            fallback_text = "Thanks, got your message."
        appended.extend(_send_only(cid, fallback_text)["messages"])

    return {"messages": appended}


def _trim_for_prompt(batch: dict[str, Any]) -> dict[str, Any]:
    outcomes = batch.get("outcomes") or []
    vouchers_processed = batch.get("vouchers_processed") or []

    # Per-voucher real-save counts: a save is real if status is in this set.
    # `duplicate` means the row already existed (re-send), don't celebrate.
    REAL_SAVE_STATUSES = {"saved", "promoted_from_draft"}
    save_counts_by_voucher: dict[str, int] = {}
    dup_counts_by_voucher: dict[str, int] = {}
    for o in outcomes:
        vid = o.get("voucher_id")
        if not vid:
            continue
        st = o.get("status") or ""
        if st in REAL_SAVE_STATUSES:
            save_counts_by_voucher[vid] = save_counts_by_voucher.get(vid, 0) + 1
        elif st == "duplicate":
            dup_counts_by_voucher[vid] = dup_counts_by_voucher.get(vid, 0) + 1

    vouchers_with_new_saves: list[dict[str, Any]] = []
    vouchers_all_duplicates: list[dict[str, Any]] = []
    vouchers_no_passports: list[dict[str, Any]] = []   # upserted but nothing routed
    for vp in vouchers_processed:
        vid = vp.get("voucher_id")
        ub = vp.get("ub_number")
        new_saves = save_counts_by_voucher.get(vid, 0)
        dups = dup_counts_by_voucher.get(vid, 0)
        entry = {"voucher_id": vid, "ub_number": ub, "new_saves": new_saves, "duplicates": dups}
        if new_saves > 0:
            vouchers_with_new_saves.append(entry)
        elif dups > 0:
            vouchers_all_duplicates.append(entry)
        else:
            vouchers_no_passports.append(entry)

    # Aggregate flags for easy prompt branching.
    total_new = sum(save_counts_by_voucher.values())
    total_dup = sum(dup_counts_by_voucher.values())
    all_duplicates = (total_new == 0 and total_dup > 0)

    primary_ub = (
        vouchers_with_new_saves[0]["ub_number"] if vouchers_with_new_saves
        else (vouchers_all_duplicates[0]["ub_number"] if vouchers_all_duplicates
              else (vouchers_processed[0].get("ub_number") if vouchers_processed else None))
    )

    return {
        "voucher_ids": batch.get("voucher_ids") or [],
        "voucher_ub": primary_ub or _first_voucher_ub(outcomes),
        # Lists for the prompt to enumerate. Only mention vouchers_with_new_saves
        # when celebrating a save; vouchers_all_duplicates for "already saved"
        # phrasing; vouchers_no_passports for "got UB-XXX, send passports when
        # ready".
        "vouchers_with_new_saves": vouchers_with_new_saves,
        "vouchers_all_duplicates": vouchers_all_duplicates,
        "vouchers_no_passports": vouchers_no_passports,
        "all_duplicates": all_duplicates,
        "new_saves_total": total_new,
        "duplicates_total": total_dup,
        "drafts_attached": (
            batch.get("drafts_attached")
            if isinstance(batch.get("drafts_attached"), int)
            else len(batch.get("drafts_attached") or [])
        ),
        "outcomes": outcomes[:8],
        "outcomes_total": len(outcomes),
        "requires_l1": batch.get("requires_l1") or [],
        "unreadable_count": batch.get("unreadable_count") or 0,
        "bad": batch.get("bad") or [],
    }


def _trim_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    sub = snapshot.get("sub_agent") or {}
    open_v = snapshot.get("open_vouchers") or []
    drafts = snapshot.get("draft_passports") or []
    return {
        "sub_agent": (
            {"name": sub.get("name"), "city": sub.get("city")}
            if isinstance(sub, dict) else None
        ),
        "open_vouchers": [
            {
                "ub": v.get("ub_number"),
                "family_head": v.get("family_head"),
                "received": v.get("received"),
                "expected": v.get("expected"),
                "missing": (v.get("missing") or [])[:5],
            }
            for v in open_v[:5]
        ],
        "draft_passports": [
            {"passport": d.get("passport"), "name": d.get("name")} for d in drafts[:5]
        ],
    }


def _first_voucher_ub(outcomes: list[dict[str, Any]]) -> str | None:
    for o in outcomes:
        ub = o.get("voucher_ub")
        if ub:
            return ub
    return None


def _send_only(conversation_id: str, text: str) -> dict[str, Any]:
    """Synthetic AIMessage+ToolMessage pair so a code-driven send still
    surfaces in the transcript for _extract_tool_calls."""
    tc_id = f"call_{abs(hash((conversation_id, text))) & 0xFFFFFFFF:x}"
    ai = AIMessage(
        content="",
        tool_calls=[{
            "id": tc_id, "name": "send_whatsapp",
            "args": {"conversation_id": conversation_id, "text": text},
        }],
    )
    tm = _run_tool_call({
        "id": tc_id, "name": "send_whatsapp",
        "args": {"conversation_id": conversation_id, "text": text},
    })
    return {"messages": [ai, tm]}


def build_agent():
    """Returns a compiled StateGraph with .invoke({"messages":[...]}, config)
    returning {"messages":[...]}. Drop-in replacement for create_react_agent."""
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent

    builder = StateGraph(GraphState)
    builder.add_node("classify", _classify_node)
    builder.add_node("extract_one", _extract_one_node)
    builder.add_node("persist", _persist_node)
    builder.add_node("recover", _recover_node)
    builder.add_node("reply", _reply_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", _route_to_extract,
        {"extract_one": "extract_one", "persist": "persist"},
    )
    builder.add_edge("extract_one", "persist")
    builder.add_conditional_edges(
        "persist", _post_persist_router,
        {"reply": "reply", "recover": "recover"},
    )
    builder.add_edge("recover", END)
    builder.add_edge("reply", END)

    return builder.compile(checkpointer=saver)
