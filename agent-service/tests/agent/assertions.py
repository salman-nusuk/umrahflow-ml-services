"""DB + trace assertion helpers.

Each `check_*` returns a list of failure strings. Empty list = pass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg


_URDU_SCRIPT_RE = re.compile(r"[؀-ۿ]")
_ROMAN_URDU_HINTS = (
    " ho gaye", " ho gaya", " ho gayi", "ke against", " kar diya",
    " mil gaya", " hain", " kar dein", " bhej dein", " shukria",
    " thoda", " abhi", " ji ", " jee ", " kr ", "ka against",
)


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------- DB queries

def fetch_outbound(conn: psycopg.Connection, cid: str) -> list[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            'SELECT id, body, status, "twilioSid", "createdAt" '
            'FROM "Message" WHERE "conversationId"=%s AND direction=%s '
            'ORDER BY "createdAt" ASC',
            (cid, "OUT"),
        )
        return list(cur.fetchall())


def fetch_inbound_processed(conn: psycopg.Connection, cid: str) -> list[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            'SELECT id, "processedAt" FROM "Message" '
            'WHERE "conversationId"=%s AND direction=%s ORDER BY "createdAt"',
            (cid, "IN"),
        )
        return list(cur.fetchall())


def fetch_latest_trace(conn: psycopg.Connection, cid: str) -> dict | None:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            'SELECT id, "toolCalls", "finalText", "durationMs", "startedAt" '
            'FROM "AgentTrace" WHERE "conversationId"=%s '
            'ORDER BY "startedAt" DESC LIMIT 1',
            (cid,),
        )
        return cur.fetchone()


def fetch_traces(conn: psycopg.Connection, cid: str) -> list[dict]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            'SELECT id, "toolCalls", "finalText", "durationMs", "startedAt" '
            'FROM "AgentTrace" WHERE "conversationId"=%s ORDER BY "startedAt" ASC',
            (cid,),
        )
        return list(cur.fetchall())


def fetch_voucher_summary(conn: psycopg.Connection, sub_agent_id: str) -> dict[str, Any]:
    """Per-voucher rollup for one sub-agent. Drafts (voucherId IS NULL) are
    counted separately by fetch_drafts_count — they're not attached to any
    voucher so they wouldn't show up here.
    Schema reminder: Passport has no `source` column; drafts are identified
    purely by `voucherId IS NULL`."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            '''SELECT v."ubNumber", v."subAgentId",
                      COUNT(p.id) AS passport_count,
                      COUNT(p.id) FILTER (WHERE p."screeningStatus"='FLAGGED') AS flagged_count
               FROM "Voucher" v LEFT JOIN "Passport" p ON p."voucherId"=v.id
               WHERE v."subAgentId"=%s GROUP BY v.id''',
            (sub_agent_id,),
        )
        return {row["ubNumber"]: row for row in cur.fetchall()}


def fetch_drafts_count(conn: psycopg.Connection, since_ms: int) -> int:
    """Count passports created since `since_ms` that aren't attached to any
    voucher. Sandbox-scoped because the test runner ensures every sandbox
    starts after this timestamp."""
    with conn.cursor() as cur:
        cur.execute(
            '''SELECT COUNT(*) FROM "Passport"
               WHERE "voucherId" IS NULL
                 AND "createdAt" > to_timestamp(%s/1000.0)''',
            (since_ms,),
        )
        row = cur.fetchone()
        return int((row[0] if isinstance(row, tuple) else row.get("count")) or 0)


# ---------------------------------------------------------------- checks

def check_db(
    conn: psycopg.Connection, sub_agent_id: str, expect: dict,
    *, since_ms: int | None = None,
) -> CheckResult:
    res = CheckResult("db")
    if not expect:
        return res
    summary = fetch_voucher_summary(conn, sub_agent_id)

    # vouchers_created — explicit list of UBs that should exist on this agent
    for ub in expect.get("vouchers_created") or []:
        if ub not in summary:
            res.failures.append(f"voucher {ub} not created (or not on this agent)")

    # passports counts
    total_passports = sum(int(r.get("passport_count") or 0) for r in summary.values())
    total_flagged = sum(int(r.get("flagged_count") or 0) for r in summary.values())
    # Drafts (voucherId IS NULL) aren't tied to a sub_agent — count them by
    # recency window. The runner passes `since_ms` from sandbox start.
    total_drafts = fetch_drafts_count(conn, since_ms) if since_ms is not None else 0

    if "passports_count" in expect and total_passports != int(expect["passports_count"]):
        res.failures.append(
            f"passports_count expected {expect['passports_count']} got {total_passports}"
        )
    if "passports_flagged" in expect and total_flagged != int(expect["passports_flagged"]):
        res.failures.append(
            f"passports_flagged expected {expect['passports_flagged']} got {total_flagged}"
        )
    if "passports_drafts" in expect and total_drafts != int(expect["passports_drafts"]):
        res.failures.append(
            f"passports_drafts expected {expect['passports_drafts']} got {total_drafts}"
        )

    # vouchers owned by other agent — sanity for cross-agent tests
    if "vouchers_owned_by_other_agent" in expect:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                'SELECT COUNT(*) AS n FROM "Voucher" '
                'WHERE "ubNumber" = ANY(%s::text[]) AND "subAgentId" <> %s',
                (expect.get("vouchers_created") or [], sub_agent_id),
            )
            n = int((cur.fetchone() or {}).get("n") or 0)
        if n != int(expect["vouchers_owned_by_other_agent"]):
            res.failures.append(
                f"vouchers_owned_by_other_agent expected "
                f"{expect['vouchers_owned_by_other_agent']} got {n}"
            )

    # requires_l1_count from trace tool_calls is checked in check_trace.
    return res


def check_reply(outbound: list[dict], expect: dict) -> CheckResult:
    res = CheckResult("reply")
    if not expect:
        return res

    if "count" in expect and len(outbound) != int(expect["count"]):
        res.failures.append(f"reply count expected {expect['count']} got {len(outbound)}")

    bodies = [(m.get("body") or "") for m in outbound]
    joined = " \n ".join(bodies)

    for sub in expect.get("must_not_contain") or []:
        if sub and sub in joined:
            res.failures.append(f"reply contains forbidden substring: {sub!r}")

    for ub in expect.get("must_mention_ubs") or []:
        if ub not in joined:
            res.failures.append(f"reply missing expected UB mention: {ub}")

    if "must_contain" in expect:
        for sub in expect["must_contain"]:
            if sub not in joined:
                res.failures.append(f"reply missing expected substring: {sub!r}")

    lang = (expect.get("language") or "").lower()
    if lang and bodies:
        if lang == "english":
            if _URDU_SCRIPT_RE.search(joined):
                res.failures.append("reply contained Urdu script but language=english")
            low = joined.lower()
            roman = [w for w in _ROMAN_URDU_HINTS if w in low]
            if roman:
                res.failures.append(
                    f"reply leaked Roman Urdu while language=english: {roman[:3]}"
                )
        elif lang == "urdu":
            if not _URDU_SCRIPT_RE.search(joined):
                res.failures.append("reply has no Urdu script but language=urdu")
        elif lang == "roman_urdu":
            low = joined.lower()
            if not any(w in low for w in _ROMAN_URDU_HINTS):
                res.failures.append("reply has no Roman Urdu markers")

    return res


def check_trace(trace: dict | None, expect: dict) -> CheckResult:
    res = CheckResult("trace")
    if not expect:
        return res
    if not trace:
        res.failures.append("no AgentTrace landed")
        return res

    raw = trace.get("toolCalls")
    calls = raw if isinstance(raw, list) else (json.loads(raw) if raw else [])
    real = [c for c in calls if c.get("name") and c.get("name") != "_meta"]
    names = [c.get("name") for c in real]

    if "tool_calls_max" in expect and len(real) > int(expect["tool_calls_max"]):
        res.failures.append(
            f"tool_calls={len(real)} exceeds max {expect['tool_calls_max']}"
        )

    if "send_whatsapp_count" in expect:
        n = sum(1 for nm in names if nm == "send_whatsapp")
        if n != int(expect["send_whatsapp_count"]):
            res.failures.append(
                f"send_whatsapp_count expected {expect['send_whatsapp_count']} got {n}"
            )

    if "must_call" in expect:
        for nm in expect["must_call"]:
            if nm not in names:
                res.failures.append(f"trace missing required tool call: {nm}")

    if "must_not_call" in expect:
        for nm in expect["must_not_call"]:
            if nm in names:
                res.failures.append(f"trace contained forbidden tool call: {nm}")

    if "requires_l1_count" in expect:
        # search for escalate_to_l1 calls as a proxy
        n = sum(1 for nm in names if nm == "escalate_to_l1")
        if n != int(expect["requires_l1_count"]):
            res.failures.append(
                f"escalate_to_l1 count expected {expect['requires_l1_count']} got {n}"
            )

    return res


def check_latency(duration_ms: int | None, expect: dict) -> CheckResult:
    res = CheckResult("latency")
    cap = expect.get("latency_ms_max")
    if cap is None or duration_ms is None:
        return res
    if int(duration_ms) > int(cap):
        res.failures.append(f"latency {duration_ms}ms exceeds {cap}ms")
    return res
