"""LangGraph tools for the UmrahFlow WhatsApp agent.

Each tool is a thin async wrapper that talks to:
  - Postgres (Prisma's DB) via psycopg
  - The OCR service on http://localhost:8001
  - Twilio's REST API (for outbound WhatsApp)
"""
import os
import uuid
import json
import base64
import datetime as dt
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import httpx
import psycopg
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool

# Sentry capture is best-effort — only initialised if a DSN is present so
# local dev / tests don't need the dep wired up. We import lazily and gate
# every capture on `_SENTRY_ENABLED` so failures here can never break a tool.
try:
    import sentry_sdk  # type: ignore
    _SENTRY_ENABLED = bool(
        os.environ.get("SENTRY_DSN_AGENT") or os.environ.get("SENTRY_DSN")
    )
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore
    _SENTRY_ENABLED = False


def _capture(e: Exception) -> None:
    """Send an exception to Sentry if configured. Never raises."""
    if _SENTRY_ENABLED and sentry_sdk is not None:
        try:
            sentry_sdk.capture_exception(e)
        except Exception:
            pass


OCR_URL = os.environ.get("OCR_SERVICE_URL", "http://127.0.0.1:8001")


def _psycopg_url(prisma_url: str) -> str:
    p = urlparse(prisma_url)
    keep = [(k, v) for k, v in parse_qsl(p.query)
            if k not in ("schema", "connection_limit", "pgbouncer", "connect_timeout")]
    return urlunparse(p._replace(query=urlencode(keep)))


DATABASE_URL = _psycopg_url(os.environ["DATABASE_URL"])
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM = os.environ["TWILIO_WHATSAPP_NUMBER"]


# Connection pool — a 6-image burst used to open ~30 fresh psycopg connections
# (one per _conn() call), and 5 concurrent bursts saturated Postgres
# max_connections=100. The pool reuses a small set of autocommit connections
# with dict_row factory, matching the prior _conn() contract exactly. Created
# once at import time so callers never re-instantiate it.
_POOL = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=2,
    max_size=20,
    kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row},
    open=True,
)


def _conn():
    """Check out an autocommit, dict_row connection from the shared pool.
    Returned object is a context manager; on __exit__ the connection is
    returned to the pool (not closed)."""
    return _POOL.connection()


# ---------------------------------------------------------------------------
# Per-turn send_whatsapp guard (B6)
# ---------------------------------------------------------------------------
# The system prompt mandates "exactly one send_whatsapp per turn". The model
# usually obeys, but the original baseline trace showed it sending 8 times for
# 3 images. This is a hard runtime guard: app.py calls reset_send_count(cid)
# before each turn; send_whatsapp checks-and-increments atomically and the
# second-and-onward calls in the same turn no-op (return an error) without
# hitting Twilio or writing to the DB.
import threading as _threading
_send_counts: dict[str, dict[str, Any]] = {}
_send_counts_lock = _threading.Lock()


def reset_send_count(conversation_id: str) -> None:
    with _send_counts_lock:
        _send_counts[conversation_id] = {"count": 0, "first_message_id": None}


def _claim_send(conversation_id: str) -> tuple[bool, str | None]:
    """Atomically check-and-claim the per-turn send slot.
    Returns (allowed, previous_message_id_if_blocked)."""
    with _send_counts_lock:
        entry = _send_counts.get(conversation_id)
        if entry is None:
            # No reset called — likely a /turn smoke test. Allow it.
            _send_counts[conversation_id] = {"count": 1, "first_message_id": None}
            return True, None
        if entry["count"] >= 1:
            return False, entry["first_message_id"]
        entry["count"] += 1
        return True, None


def _record_first_send(conversation_id: str, message_id: str) -> None:
    """Record the first successful send so a blocked second call can reference it."""
    with _send_counts_lock:
        entry = _send_counts.get(conversation_id)
        if entry is not None and entry["first_message_id"] is None:
            entry["first_message_id"] = message_id


# ---------------------------------------------------------------------------
# 1. classify_media — cheap GPT-5.4 Nano vision call
# ---------------------------------------------------------------------------

@tool
def classify_media(message_id: str, media_idx: int = 0) -> str:
    """Classify a WhatsApp media attachment as one of:
    'passport', 'voucher', 'id_card', 'payment_proof', 'other', 'unreadable'.

    Args:
        message_id: The Message.id of the inbound WhatsApp message that has media.
        media_idx: Which media item (0-indexed) on the message. Default 0.
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute('SELECT media FROM "Message" WHERE id = %s', (message_id,))
        row = cur.fetchone()
        if not row or not row.get("media"):
            return "no media on message"
        media = row["media"][media_idx] if isinstance(row["media"], list) else row["media"]

    # PDF short-circuit: GPT-vision can't classify PDFs directly. Return "pdf"
    # so the agent calls extract_pdf, which renders + classifies each page.
    content_type = (media.get("contentType") or "").lower()
    if "pdf" in content_type:
        return "pdf"

    # fetch the image (Twilio media requires Basic Auth)
    try:
        with httpx.Client(timeout=30, auth=(TWILIO_SID, TWILIO_TOKEN), follow_redirects=True) as cli:
            r = cli.get(media["url"])
            r.raise_for_status()
            img_b64 = base64.b64encode(r.content).decode()
    except Exception as e:
        _capture(e)
        return f"fetch failed: {e}"

    payload = {
        "model": "gpt-5.4-nano",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Classify this image. Reply with one word, lowercase: "
                    "passport (Pakistani passport bio page), "
                    "voucher (visa/hotel/booking voucher), "
                    "id_card (CNIC or other ID), "
                    "payment_proof (bank slip, receipt, transfer screenshot), "
                    "other, or unreadable."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        "max_completion_tokens": 8,
    }
    with httpx.Client(timeout=60) as cli:
        r = cli.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
        )
    if r.status_code != 200:
        return f"classifier error: {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"].strip().lower().strip(".")


# ---------------------------------------------------------------------------
# 2. extract_passport — calls our existing OCR service
# ---------------------------------------------------------------------------

@tool
def extract_passport(message_id: str, media_idx: int = 0) -> dict[str, Any]:
    """Run OCR on a Pakistani passport image to extract MRZ and visual-zone fields.
    Returns JSON with mrz_status, mrz fields, viz fields. Use this only AFTER
    classify_media has confirmed the image is a passport.

    Args:
        message_id: Message.id holding the passport image.
        media_idx: Which media item (0-indexed). Default 0.
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute('SELECT media FROM "Message" WHERE id = %s', (message_id,))
        row = cur.fetchone()
        if not row or not row.get("media"):
            return {"error": "no media"}
        media = row["media"][media_idx] if isinstance(row["media"], list) else row["media"]

    try:
        with httpx.Client(timeout=120, auth=(TWILIO_SID, TWILIO_TOKEN), follow_redirects=True) as cli:
            blob = cli.get(media["url"])
            blob.raise_for_status()
            r = httpx.post(f"{OCR_URL}/ocr", files={"file": ("img.jpg", blob.content)}, timeout=120)
        result = r.json()
    except Exception as e:
        _capture(e)
        return {"error": str(e)}

    # Fat return (T2.5): enrich with blacklist + voucher-match preview when
    # the OCR yielded a passport_number. Saves the model two follow-up tool
    # calls per passport since it can decide create vs draft vs escalate
    # immediately. Best-effort — failures here don't fail the OCR call.
    primary = (result.get("primary") or {}) if isinstance(result, dict) else {}
    pno = (primary.get("passport_number") or "").strip().upper()
    if pno:
        try:
            with _conn() as c, c.cursor() as cur:
                cur.execute('SELECT * FROM "Blacklist" WHERE "passportNumber" = %s', (pno,))
                bl = cur.fetchone()
                cur.execute(
                    '''SELECT id, "ubNumber", "familyHead", "subAgentId" FROM "Voucher"
                       WHERE "expectedMutamers" @> %s::jsonb
                       ORDER BY "createdAt" DESC LIMIT 5''',
                    (json.dumps([{"passport": pno}]),),
                )
                matches = cur.fetchall()
            result["blacklist_preview"] = bl if bl else None
            if not matches:
                result["match_preview"] = {"match": None}
            elif len(matches) == 1:
                result["match_preview"] = {"match": matches[0]}
            else:
                result["match_preview"] = {"matches": matches}
        except Exception as e:
            _capture(e)
            # Silent — model can still call check_blacklist / match explicitly.
            pass

    return result


# ---------------------------------------------------------------------------
# 3-5. Database lookups — agent, voucher, blacklist
# ---------------------------------------------------------------------------

@tool
def lookup_agent(phone: str) -> dict[str, Any] | None:
    """Look up a SubAgent (travel agency) by phone number. Returns agent name,
    contact, city, status, credit limit, and ID. Returns null if not found."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            'SELECT id, name, "contactName", phone, city, status, "creditLimit" '
            'FROM "SubAgent" WHERE phone = %s',
            (phone,),
        )
        row = cur.fetchone()
    return row


@tool
def lookup_voucher(ub_number: str) -> dict[str, Any] | None:
    """Look up a Voucher by its UB number (e.g. UB-169500). Returns voucher
    metadata including current status, total amount, amount received, and the
    sub-agent ID."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            'SELECT id, "ubNumber", status, "totalAmount", "amountReceived", "subAgentId" '
            'FROM "Voucher" WHERE "ubNumber" = %s',
            (ub_number,),
        )
        row = cur.fetchone()
    return row


@tool
def check_blacklist(passport_number: str) -> dict[str, Any] | None:
    """Check if a passport number is on the blacklist. Returns the blacklist
    entry (with reason and date) if found, otherwise null."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            'SELECT id, "passportNumber", reason, "createdAt" '
            'FROM "Blacklist" WHERE "passportNumber" = %s',
            (passport_number,),
        )
        row = cur.fetchone()
    return row


# ---------------------------------------------------------------------------
# 6. create_passport — silent save of OCR result
# ---------------------------------------------------------------------------

def _passport_history(cur, passport_number: str) -> list[dict]:
    """Return prior Passport rows still considered 'active' (not yet ISSUED or
    REJECTED). Used to detect cross-agent or duplicate submissions."""
    cur.execute(
        '''SELECT p.id, p."voucherId", p."givenNames", p.surname,
                  v."ubNumber", v.status AS voucher_status, v."subAgentId",
                  v."returnDate", a.name AS agent_name
           FROM "Passport" p
           LEFT JOIN "Voucher" v ON v.id = p."voucherId"
           LEFT JOIN "SubAgent" a ON a.id = v."subAgentId"
           WHERE p."passportNumber" = %s
             AND (v.status IS NULL OR v.status NOT IN ('ISSUED','REJECTED'))''',
        (passport_number,),
    )
    return cur.fetchall()


def _scan_url_for_message(cur, message_id: str | None, media_idx: int = 0) -> str | None:
    """Return the dashboard-side proxy URL for the message's media so the OCR
    review UI can render the original scan. The raw Twilio URL requires basic
    auth; the /api/media/[messageId]/[idx] route proxies it with credentials
    server-side."""
    if not message_id:
        return None
    cur.execute('SELECT media, "mediaUrl" FROM "Message" WHERE id = %s', (message_id,))
    row = cur.fetchone()
    if not row:
        return None
    media = row.get("media")
    has_media = (isinstance(media, list) and len(media) > media_idx) or row.get("mediaUrl")
    if not has_media:
        return None
    return f"/api/media/{message_id}/{media_idx}"


@tool
def create_passport(
    voucher_id: str,
    surname: str,
    given_names: str,
    passport_number: str,
    date_of_birth: str,
    sex: str,
    expiry_date: str,
    cnic: str | None = None,
    place_of_birth: str | None = None,
    booklet_number: str | None = None,
    source_message_id: str | None = None,
    verified: bool = True,
    verification_notes: str | None = None,
) -> dict[str, Any]:
    """Save a passport against a voucher. Idempotent: if the same passport is
    already attached to this voucher, returns the existing row without writing.
    Cross-agent aware: if the same passport is currently active on a DIFFERENT
    sub-agent's voucher, returns {conflict: 'cross_agent', existing: ...} for
    the agent to escalate. Otherwise writes and returns {id, voucher_id}.
    Required: voucher_id, surname, given_names, passport_number, date_of_birth,
    sex (M/F), expiry_date (YYYY-MM-DD).
    Pass source_message_id (the inbound Message.id holding the passport image)
    so the OCR review UI can render the original scan.
    `verified` and `verification_notes` are MRZ cross-check outcomes from
    the OCR pipeline. Per current policy MRZ is informational only — the
    note is recorded in screeningNotes for L1 to inspect, but it does NOT
    auto-flag the row. Blacklist hits and cross-agent conflicts are the
    only triggers that set screeningStatus='FLAGGED'."""
    import uuid
    initial_status = "PENDING"
    notes = verification_notes  # always retained; never used as a flag trigger
    with _conn() as c, c.cursor() as cur:
        scan_url = _scan_url_for_message(cur, source_message_id)
        history = _passport_history(cur, passport_number)

        # idempotent: already attached to this voucher
        for row in history:
            if row["voucherId"] == voucher_id:
                return {
                    "id": row["id"],
                    "voucher_id": voucher_id,
                    "duplicate": True,
                    "note": "already saved on this voucher",
                }

        # find target voucher's sub-agent for cross-agent comparison
        cur.execute('SELECT "subAgentId" FROM "Voucher" WHERE id = %s', (voucher_id,))
        target = cur.fetchone()
        if not target:
            return {"error": "voucher not found"}
        target_agent = target["subAgentId"]

        # cross-agent active conflict
        for row in history:
            if row["voucherId"] and row["subAgentId"] and row["subAgentId"] != target_agent:
                return {
                    "conflict": "cross_agent",
                    "existing": {
                        "ub_number": row["ubNumber"],
                        "agent": row["agent_name"],
                        "status": row["voucher_status"],
                    },
                    "note": "passport currently active on another agent's voucher; needs L1 review",
                }

        # promote a draft (voucherId NULL) for the same passport_number, if any
        for row in history:
            if row["voucherId"] is None:
                cur.execute(
                    '''UPDATE "Passport" SET "voucherId" = %s,
                       "subAgentId" = COALESCE(%s, "subAgentId"),
                       surname = COALESCE(%s, surname),
                       "givenNames" = COALESCE(%s, "givenNames"),
                       "dateOfBirth" = COALESCE(%s::date, "dateOfBirth"),
                       gender = COALESCE(%s, gender),
                       "expiryDate" = COALESCE(%s::date, "expiryDate"),
                       cnic = COALESCE(%s, cnic),
                       "placeOfBirth" = COALESCE(%s, "placeOfBirth"),
                       "bookletNumber" = COALESCE(%s, "bookletNumber"),
                       nationality = COALESCE(nationality, 'PAK'),
                       "issuingCountry" = COALESCE("issuingCountry", 'PAK'),
                       "scanUrl" = COALESCE("scanUrl", %s),
                       "screeningNotes" = COALESCE(%s, "screeningNotes"),
                       "updatedAt" = NOW()
                       WHERE id = %s
                       RETURNING id''',
                    (voucher_id, target_agent, surname, given_names, date_of_birth, sex, expiry_date,
                     cnic, place_of_birth, booklet_number, scan_url,
                     notes, row["id"]),
                )
                return {
                    "id": cur.fetchone()["id"],
                    "voucher_id": voucher_id,
                    "promoted_from_draft": True,
                    "screening_status": initial_status,
                }

        # fresh insert — stamp subAgentId from the voucher's owner so
        # ownership/attribution is correct from row 1 (B2B tenant scoping).
        cur.execute(
            '''INSERT INTO "Passport"
               (id, "voucherId", "subAgentId", surname, "givenNames", "passportNumber",
                "dateOfBirth", gender, "expiryDate", cnic, "placeOfBirth", "bookletNumber",
                nationality, "issuingCountry", "scanUrl", "screeningNotes",
                "screeningStatus", "nusukStatus", "visaStatus",
                "uploadSource", "createdAt", "updatedAt")
               VALUES (%s, %s, %s, %s, %s, %s, %s::date, %s,
                       %s::date, %s, %s, %s,
                       'PAK', 'PAK', %s, %s,
                       %s::"ScreeningStatus", 'NOT_SUBMITTED', 'PENDING',
                       'WHATSAPP'::"UploadSource", NOW(), NOW())
               RETURNING id''',
            (str(uuid.uuid4()), voucher_id, target_agent, surname, given_names, passport_number,
             date_of_birth, sex, expiry_date, cnic, place_of_birth, booklet_number, scan_url,
             notes, initial_status),
        )
        return {
            "id": cur.fetchone()["id"],
            "voucher_id": voucher_id,
            "screening_status": initial_status,
        }


@tool
def create_passport_draft(
    surname: str,
    given_names: str,
    passport_number: str,
    date_of_birth: str,
    sex: str,
    expiry_date: str,
    cnic: str | None = None,
    place_of_birth: str | None = None,
    booklet_number: str | None = None,
    source_message_id: str | None = None,
    verified: bool = True,
    verification_notes: str | None = None,
) -> dict[str, Any]:
    """Save a passport as a DRAFT (no voucher attached yet). Use this when OCR
    succeeds but match_passport_to_voucher returns no match — the user hasn't
    sent the voucher PDF yet. The draft will auto-attach when the matching
    voucher arrives. Idempotent: same passport_number returns the existing draft.
    Pass source_message_id (the inbound Message.id holding the passport image)
    so the OCR review UI can render the original scan.
    `verified` and `verification_notes` are MRZ cross-check outcomes. Per
    current policy MRZ is informational only — the note is recorded in
    screeningNotes for L1 to inspect, but it does NOT auto-flag the draft."""
    import uuid
    initial_status = "PENDING"
    notes = verification_notes
    with _conn() as c, c.cursor() as cur:
        scan_url = _scan_url_for_message(cur, source_message_id)
        cur.execute(
            'SELECT id FROM "Passport" WHERE "passportNumber" = %s AND "voucherId" IS NULL',
            (passport_number,),
        )
        existing = cur.fetchone()
        if existing:
            if scan_url or notes:
                cur.execute(
                    '''UPDATE "Passport" SET "scanUrl" = COALESCE("scanUrl", %s),
                       nationality = COALESCE(nationality, 'PAK'),
                       "issuingCountry" = COALESCE("issuingCountry", 'PAK'),
                       "screeningNotes" = COALESCE(%s, "screeningNotes"),
                       "updatedAt" = NOW()
                       WHERE id = %s''',
                    (scan_url, notes, existing["id"]),
                )
            return {"id": existing["id"], "duplicate": True}
        cur.execute(
            '''INSERT INTO "Passport"
               (id, "voucherId", surname, "givenNames", "passportNumber",
                "dateOfBirth", gender, "expiryDate", cnic, "placeOfBirth", "bookletNumber",
                nationality, "issuingCountry", "scanUrl", "screeningNotes",
                "screeningStatus", "nusukStatus", "visaStatus",
                "uploadSource", "createdAt", "updatedAt")
               VALUES (%s, NULL, %s, %s, %s, %s::date, %s,
                       %s::date, %s, %s, %s,
                       'PAK', 'PAK', %s, %s,
                       %s::"ScreeningStatus", 'NOT_SUBMITTED', 'PENDING',
                       'WHATSAPP'::"UploadSource", NOW(), NOW())
               RETURNING id''',
            (str(uuid.uuid4()), surname, given_names, passport_number,
             date_of_birth, sex, expiry_date, cnic, place_of_birth, booklet_number, scan_url,
             notes, initial_status),
        )
        return {
            "id": cur.fetchone()["id"],
            "draft": True,
            "screening_status": initial_status,
        }


@tool
def scan_drafts_for_voucher(voucher_id: str) -> dict[str, Any]:
    """After upserting a voucher, scan existing draft passports (voucherId IS NULL)
    and auto-attach any whose passport_number is in this voucher's expectedMutamers.
    Returns {attached: [...], cross_agent_skipped: [...]} — the second list holds
    passport_numbers that have prior history on a DIFFERENT sub-agent's voucher
    and must NOT be silently re-attached (B2B data leak)."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            'SELECT "expectedMutamers", "subAgentId" FROM "Voucher" WHERE id = %s', (voucher_id,)
        )
        row = cur.fetchone()
        if not row or not row["expectedMutamers"]:
            return {"attached": [], "cross_agent_skipped": []}
        passport_numbers = [m.get("passport") for m in row["expectedMutamers"] if m.get("passport")]
        if not passport_numbers:
            return {"attached": [], "cross_agent_skipped": []}

        target_agent = row["subAgentId"]
        # Bug #3: filter out drafts that have a same-pno history on a different
        # sub-agent's voucher. Without this, agent A's draft would auto-attach
        # to agent B's UB the moment B's voucher arrives — cross-tenant leak.
        # Acceptable signal: ANY prior Passport row for the same pno with a
        # non-null voucherId belonging to a sub_agent_id != target_agent.
        cross_agent_skipped: list[str] = []
        safe_numbers: list[str] = []
        if target_agent:
            for pno in passport_numbers:
                cur.execute(
                    '''SELECT 1 FROM "Passport" p
                       JOIN "Voucher" v ON v.id = p."voucherId"
                       WHERE p."passportNumber" = %s
                         AND v."subAgentId" IS NOT NULL
                         AND v."subAgentId" <> %s
                       LIMIT 1''',
                    (pno, target_agent),
                )
                if cur.fetchone():
                    cross_agent_skipped.append(pno)
                else:
                    safe_numbers.append(pno)
        else:
            # no target agent known — fall back to old behaviour, can't compare.
            safe_numbers = passport_numbers

        if not safe_numbers:
            return {"attached": [], "cross_agent_skipped": cross_agent_skipped}
        cur.execute(
            '''UPDATE "Passport"
               SET "voucherId" = %s,
                   "subAgentId" = COALESCE(%s, "subAgentId"),
                   "updatedAt" = NOW()
               WHERE "voucherId" IS NULL AND "passportNumber" = ANY(%s)
               RETURNING id, "passportNumber", "givenNames", surname''',
            (voucher_id, target_agent, safe_numbers),
        )
        return {"attached": cur.fetchall(), "cross_agent_skipped": cross_agent_skipped}


@tool
def conversation_state(conversation_id: str) -> dict[str, Any]:
    """Snapshot of the current state of this conversation, taken at the start
    of every turn. Returns sub-agent (if linked), open vouchers (UB number,
    family head, received vs expected count, list of missing passport numbers),
    draft passports awaiting a voucher, and contact phone. The agent should
    use this as ground truth before deciding what to do."""
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            'SELECT id, phone, "subAgentId" FROM "Conversation" WHERE id = %s',
            (conversation_id,),
        )
        convo = cur.fetchone()
        if not convo:
            return {"error": "conversation not found"}

        sub_agent = None
        if convo["subAgentId"]:
            cur.execute(
                'SELECT id, name, "contactName", phone, status FROM "SubAgent" WHERE id = %s',
                (convo["subAgentId"],),
            )
            sub_agent = cur.fetchone()

        open_vouchers = []
        if sub_agent:
            cur.execute(
                '''SELECT v.id, v."ubNumber", v."familyHead", v."expectedMutamers",
                          v.status, v."createdAt"
                   FROM "Voucher" v
                   WHERE v."subAgentId" = %s
                     AND v.status NOT IN ('ISSUED','REJECTED')
                   ORDER BY v."createdAt" DESC LIMIT 20''',
                (sub_agent["id"],),
            )
            vouchers = cur.fetchall()
            for v in vouchers:
                expected = v["expectedMutamers"] or []
                expected_nums = [m.get("passport") for m in expected if m.get("passport")]
                cur.execute(
                    'SELECT "passportNumber", "givenNames", surname FROM "Passport" WHERE "voucherId" = %s',
                    (v["id"],),
                )
                attached = cur.fetchall()
                attached_nums = {p["passportNumber"] for p in attached}
                missing = [m for m in expected if m.get("passport") and m.get("passport") not in attached_nums]
                open_vouchers.append({
                    "ub_number": v["ubNumber"],
                    "family_head": v["familyHead"],
                    "voucher_id": v["id"],
                    "status": v["status"],
                    "received": len(attached),
                    "expected": len(expected_nums),
                    "missing": [{"passport": m.get("passport"), "name": m.get("name")} for m in missing],
                })

        # Bug #4: scope drafts so we don't leak competitor data into the
        # reply prompt. Show only drafts that EITHER (a) have no history of
        # ever being attached to ANY agent's voucher (truly orphan, can move
        # to anyone), OR (b) were created in THIS conversation's own message
        # history (linked via Passport.scanUrl -> Message.id -> conversationId).
        # Everything else is dropped.
        cur.execute(
            '''SELECT id, "passportNumber", "givenNames", surname, "scanUrl", "createdAt"
               FROM "Passport" WHERE "voucherId" IS NULL ORDER BY "createdAt" DESC LIMIT 50'''
        )
        candidate_drafts = cur.fetchall()
        drafts: list[dict] = []
        for d in candidate_drafts:
            pno = d["passportNumber"]
            # (a) truly orphan: no prior attached row for this pno on anyone's voucher
            cur.execute(
                '''SELECT 1 FROM "Passport"
                   WHERE "passportNumber" = %s AND "voucherId" IS NOT NULL LIMIT 1''',
                (pno,),
            )
            has_history = cur.fetchone() is not None

            in_this_convo = False
            scan_url = d.get("scanUrl") or ""
            # scan_url shape: /api/media/<messageId>/<idx>
            if scan_url.startswith("/api/media/"):
                parts = scan_url.split("/")
                msg_id = parts[3] if len(parts) > 3 else None
                if msg_id:
                    cur.execute(
                        'SELECT "conversationId" FROM "Message" WHERE id = %s',
                        (msg_id,),
                    )
                    mrow = cur.fetchone()
                    if mrow and mrow["conversationId"] == convo["id"]:
                        in_this_convo = True

            if (not has_history) or in_this_convo:
                drafts.append(d)
            if len(drafts) >= 20:
                break

    return {
        "conversation_id": convo["id"],
        "phone": convo["phone"],
        "sub_agent": sub_agent,
        "open_vouchers": open_vouchers,
        "draft_passports": [
            {"passport": d["passportNumber"], "name": f"{d['givenNames'] or ''} {d['surname'] or ''}".strip()}
            for d in drafts
        ],
    }


# ---------------------------------------------------------------------------
# 7. send_whatsapp — outbound reply
# ---------------------------------------------------------------------------

@tool
def send_whatsapp(conversation_id: str, text: str) -> dict[str, Any]:
    """Send a WhatsApp message to the agent on this conversation. Writes the
    message to the DB and dispatches via Twilio. Reply in plain English, concise.

    Hard guard: at most one successful send per turn. A second call in the same
    turn returns {"error":"already_sent_this_turn", ...} without invoking Twilio
    or writing to the DB."""
    # Hard send-once guard (B6). Claim the slot before doing any work; if
    # already claimed by an earlier call in this turn, no-op.
    allowed, prev_id = _claim_send(conversation_id)
    if not allowed:
        return {
            "error": "already_sent_this_turn",
            "previous_message_id": prev_id,
            "hint": "You already replied this turn. Do not call send_whatsapp again.",
        }
    # Step 1: look up phone and write a QUEUED outbound row BEFORE hitting Twilio.
    # If we crash between the Twilio call and the post-call DB write, this row is
    # the receipt — replay logic can see the intent and skip a duplicate send.
    with _conn() as c, c.cursor() as cur:
        cur.execute('SELECT phone FROM "Conversation" WHERE id = %s', (conversation_id,))
        row = cur.fetchone()
        if not row:
            return {"error": "conversation not found"}
        phone = row["phone"]

        cur.execute(
            '''INSERT INTO "Message"
               (id, "conversationId", direction, body, status, "createdAt")
               VALUES (gen_random_uuid()::text, %s, 'OUT', %s, 'QUEUED', NOW())
               RETURNING id''',
            (conversation_id, text),
        )
        msg_id = cur.fetchone()["id"]

    # Step 2: actually send (or skip in TEST_MODE).
    twilio_sid = None
    twilio_error = None
    if os.environ.get("TEST_MODE", "").strip() in {"1", "true", "yes", "on"}:
        twilio_sid = f"TEST-{uuid.uuid4().hex[:16]}"
    else:
        try:
            with httpx.Client(timeout=30, auth=(TWILIO_SID, TWILIO_TOKEN)) as cli:
                r = cli.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                    data={"From": TWILIO_FROM, "To": f"whatsapp:{phone}", "Body": text},
                )
            if r.status_code in (200, 201):
                twilio_sid = r.json().get("sid")
            else:
                twilio_error = f"twilio {r.status_code}: {r.text[:200]}"
        except Exception as e:
            _capture(e)
            twilio_error = f"twilio exception: {str(e)[:200]}"

    # Step 3: finalize the row. Mark FAILED on Twilio error so it's visible in
    # the inbox UI and won't be retried automatically.
    with _conn() as c, c.cursor() as cur:
        if twilio_sid:
            cur.execute(
                '''UPDATE "Message"
                   SET "twilioSid" = %s, status = 'SENT'
                   WHERE id = %s''',
                (twilio_sid, msg_id),
            )
            cur.execute(
                'UPDATE "Conversation" SET "lastMessageAt" = NOW() WHERE id = %s',
                (conversation_id,),
            )
        else:
            cur.execute(
                'UPDATE "Message" SET status = \'FAILED\' WHERE id = %s',
                (msg_id,),
            )

    if twilio_error:
        return {"error": twilio_error, "message_id": msg_id}
    _record_first_send(conversation_id, msg_id)
    return {"message_id": msg_id, "twilio_sid": twilio_sid}


# ---------------------------------------------------------------------------
# 8. escalate_to_l1 — flips bot off, sets reason
# ---------------------------------------------------------------------------

@tool
def escalate_to_l1(conversation_id: str, reason: str) -> dict[str, Any]:
    """Hand this conversation off to a human L1 operator. Disables the bot on
    this conversation and records why. Use when uncertain, when the user explicitly
    asks for a human, or when something goes wrong (blacklist hit, expired
    passport, payment dispute, complaint, off-topic emergency).

    Args:
        conversation_id: The Conversation.id.
        reason: Short reason for escalation, shown to L1.
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            '''UPDATE "Conversation"
               SET "botEnabled" = false,
                   "escalationReason" = %s,
                   "escalatedAt" = NOW()
               WHERE id = %s''',
            (reason, conversation_id),
        )
    return {"escalated": True, "reason": reason}


# ---------------------------------------------------------------------------
# 9. extract_voucher — pdfplumber + GPT-5.4 Nano structured parse
# ---------------------------------------------------------------------------

VOUCHER_PARSE_PROMPT = """Parse this Pakistani Umrah hotel-voucher into JSON.

Return ONLY a JSON object with these keys (use null when not present):
{
  "ub_number": "UB-XXXXXX",
  "agency_name": "header company name",
  "voucher_date": "YYYY-MM-DD",
  "package_nights": <integer>,
  "family_head": "name",
  "pax_adults": <int>,
  "pax_children": <int>,
  "pax_infants": <int>,
  "mutamers": [
    {"passport": "EK1234567", "name": "FULL NAME", "gender": "M|F", "pax_type": "Adult|Child|Infant"}
  ],
  "airline": "carrier name e.g. Saudia, Emirates, AirSial",
  "departure_city": "ISB|LHE|KHI|...",
  "outbound_flight": "ER-123",
  "outbound_date": "YYYY-MM-DD",
  "return_flight": "ER-456",
  "return_date": "YYYY-MM-DD",
  "hotel_name": "string",
  "hotel_city": "Makkah|Madinah",
  "checkin_date": "YYYY-MM-DD",
  "checkout_date": "YYYY-MM-DD",
  "room_type": "Quad|Triple|Double|Quint|Twin|Single",
  "room_count": <integer>
}

PASSPORT NUMBER RULES (CRITICAL — most common error source):
- Pakistani passport numbers are typically 9 characters: 2 letters + 7 digits
  (e.g. CD8022311, GF9211711, AB1234567). Older booklets are 8 chars
  (1 letter + 7 digits). They are NEVER fewer than 8 characters.
- The Mutamers table has tightly-spaced columns. Read each character one at a
  time. Do not collapse repeated digits — "22" is two characters, not one.
- After extracting each passport number, COUNT the characters. If you got
  fewer than 8, you missed a digit — re-read the column.
- Letters are always upper-case. Digits 0-9. No spaces, no dashes.

Convert dates from DD-MM-YY or DD/MM/YY to YYYY-MM-DD (assume 20YY for 2-digit years).
Do NOT include any text outside the JSON."""


def _voucher_from_text(text: str) -> dict[str, Any]:
    """Send extracted text to GPT for structured voucher parsing."""
    payload = {
        "model": "gpt-5.4-nano",
        "messages": [{
            "role": "user",
            "content": VOUCHER_PARSE_PROMPT + "\n\nVOUCHER TEXT:\n" + text[:6000],
        }],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1500,
    }
    with httpx.Client(timeout=60) as cli:
        r = cli.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
        )
    if r.status_code != 200:
        return {"error": f"openai {r.status_code}: {r.text[:200]}"}
    return json.loads(r.json()["choices"][0]["message"]["content"])


VOUCHER_VISION_MODEL = os.environ.get("OPENAI_VOUCHER_MODEL", "gpt-5.4-mini")


def _voucher_from_image(img_bytes: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """Vision-parse a voucher (image or rendered PDF page). Uses the higher-
    tier vision model — passport numbers in voucher tables are tiny and
    tightly-spaced, so accuracy matters more than cost."""
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": VOUCHER_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VOUCHER_PARSE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        # high detail: ensures the model gets full-resolution
                        # patches on small table text instead of a downsampled
                        # thumbnail.
                        "detail": "high",
                    },
                },
            ],
        }],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 1500,
    }
    with httpx.Client(timeout=180) as cli:
        r = cli.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
        )
    if r.status_code != 200:
        return {"error": f"openai {r.status_code}: {r.text[:200]}"}
    return json.loads(r.json()["choices"][0]["message"]["content"])


def _render_pdf_first_page_jpeg(pdf_bytes: bytes) -> bytes:
    """Render page 1 of a PDF to JPEG bytes via pypdfium2."""
    import io
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        page = pdf[0]
        pil = page.render(scale=200 / 72.0).to_pil().convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=92)
        return buf.getvalue()
    finally:
        pdf.close()


@tool
def extract_voucher(message_id: str, media_idx: int = 0) -> dict[str, Any]:
    """Parse a hotel-voucher attachment into structured fields (ub_number,
    family_head, mutamers list, etc.). Handles both PDFs and image vouchers
    (JPEG/PNG screenshots).

    Uses GPT-vision as the primary extractor (renders the PDF page first if
    needed) — text-only PDF extraction silently drops characters in tight
    table fonts (especially passport numbers), so vision is the safer default.
    pdfplumber text extraction is only used as a fallback if vision fails.

    Use this only AFTER classify_media confirms the media is a 'voucher' (or
    after extract_pdf classifies a page as 'voucher')."""
    import pdfplumber, io

    with _conn() as c, c.cursor() as cur:
        cur.execute('SELECT media FROM "Message" WHERE id = %s', (message_id,))
        row = cur.fetchone()
        if not row or not row.get("media"):
            return {"error": "no media"}
        media = row["media"][media_idx] if isinstance(row["media"], list) else row["media"]

    content_type = (media.get("contentType") or "").lower()
    try:
        with httpx.Client(timeout=60, auth=(TWILIO_SID, TWILIO_TOKEN), follow_redirects=True) as cli:
            r = cli.get(media["url"])
            r.raise_for_status()
            blob = r.content
    except Exception as e:
        _capture(e)
        return {"error": f"fetch failed: {e}"}

    is_pdf = "pdf" in content_type or blob[:4] == b"%PDF"

    # Tag the parsed result with provenance so upsert_voucher can persist
    # scanUrls + sourceKind for the unified review surface.
    scan_url = f"/api/media/{message_id}/{media_idx}"
    source_kind = "PDF" if is_pdf else "IMAGE"

    def _tag(d: dict[str, Any]) -> dict[str, Any]:
        if isinstance(d, dict) and "error" not in d:
            d["_scan_url"] = scan_url
            d["_source_kind"] = source_kind
        return d

    # Vision-first path. For PDFs, render page 1 to JPEG and feed to vision.
    # For images, feed directly. This avoids the silent character-drop bug in
    # pdfplumber text extraction on tightly-spaced table fonts.
    try:
        if is_pdf:
            img_bytes = _render_pdf_first_page_jpeg(blob)
            return _tag(_voucher_from_image(img_bytes, "image/jpeg"))
        return _tag(_voucher_from_image(blob, content_type or "image/jpeg"))
    except Exception as vision_err:
        _capture(vision_err)
        # Vision failed — try pdfplumber text as a last resort for PDFs.
        if not is_pdf:
            return {"error": f"vision parse failed: {vision_err}"}
        try:
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if text.strip():
                return _tag(_voucher_from_text(text))
            return {"error": f"vision failed and PDF has no text: {vision_err}"}
        except Exception as text_err:
            _capture(text_err)
            return {"error": f"vision failed ({vision_err}); pdfplumber failed ({text_err})"}


# ---------------------------------------------------------------------------
# 10. upsert_voucher — save parsed voucher to DB
# ---------------------------------------------------------------------------

@tool
def upsert_voucher(
    ub_number: str,
    family_head: str | None = None,
    agency_name: str | None = None,
    expected_mutamers: list[dict] | None = None,
    departure_city: str | None = None,
    arrival_city: str | None = None,
    sub_agent_id: str | None = None,
    scan_url: str | None = None,
    source_kind: str | None = None,
    outbound_flight: str | None = None,
    outbound_date: str | None = None,
    return_date: str | None = None,
    airline: str | None = None,
    hotel_name: str | None = None,
    hotel_city: str | None = None,
    checkin_date: str | None = None,
    checkout_date: str | None = None,
    room_type: str | None = None,
    room_count: int | None = None,
) -> dict[str, Any]:
    """Create or update a Voucher row by UB number. Pass expected_mutamers as
    a JSON list of {passport, name, gender, pax_type} from the parsed voucher.
    Returns the voucher id. Used after extract_voucher succeeds.

    totalAmount is auto-computed as agent.pricePerPassport × len(expected_mutamers).
    On update, total is recomputed only when expected_mutamers grows or when the
    sub-agent gets reassigned, so manual L2 edits to totalAmount aren't blown away."""
    import re
    import uuid
    expected_json = json.dumps(expected_mutamers) if expected_mutamers else None
    pax_count = len(expected_mutamers) if expected_mutamers else 0

    # Derive airline IATA-ish prefix from flight code when the prompt didn't
    # extract a separate airline name. "ER-123" / "SV456" → "ER" / "SV".
    if not airline and outbound_flight:
        m = re.match(r"^\s*([A-Z]{1,3})[\s\-/]?\d+", outbound_flight.upper())
        if m:
            airline = m.group(1)

    with _conn() as c, c.cursor() as cur:
        # Look up the agent's per-passport rate when known.
        rate = None
        if sub_agent_id:
            cur.execute('SELECT "pricePerPassport" FROM "SubAgent" WHERE id = %s', (sub_agent_id,))
            r = cur.fetchone()
            rate = r["pricePerPassport"] if r else None
        computed_total = (rate * pax_count) if (rate is not None and pax_count > 0) else None

        cur.execute('SELECT id, "totalAmount", "subAgentId", "expectedMutamers" FROM "Voucher" WHERE "ubNumber" = %s', (ub_number,))
        existing = cur.fetchone()
        if existing:
            # Bug #2: cross-agent hijack guard. The previous UPDATE used
            # COALESCE(%s, "subAgentId") which let agent B silently overwrite
            # agent A's ownership of an existing UB. Refuse to touch the row
            # when an existing non-null subAgentId differs from the incoming
            # one — caller (finalize_passport_batch) treats this as a per-
            # voucher failure and surfaces it to L1.
            if (
                sub_agent_id
                and existing["subAgentId"]
                and existing["subAgentId"] != sub_agent_id
            ):
                return {
                    "error": "cross_agent_conflict",
                    "existing_agent": existing["subAgentId"],
                    "ub_number": ub_number,
                    "voucher_id": existing["id"],
                }
            # only overwrite total if (a) it's still 0 (untouched), or (b) the
            # passenger count grew, or (c) the agent assignment changed.
            prior_total = existing["totalAmount"]
            prior_expected = existing["expectedMutamers"] or []
            prior_count = len(prior_expected) if isinstance(prior_expected, list) else 0
            agent_changed = sub_agent_id and existing["subAgentId"] != sub_agent_id
            should_overwrite = (
                computed_total is not None
                and (
                    (prior_total is None or float(prior_total) == 0.0)
                    or pax_count > prior_count
                    or agent_changed
                )
            )
            # On update we APPEND scan_url to scanUrls (multi-attach over time
            # is normal — same UB re-sent as a corrected PDF). screeningStatus
            # only flips back to PENDING if a fresh scan came in, otherwise the
            # existing review state is preserved.
            cur.execute(
                '''UPDATE "Voucher" SET
                   "familyHead" = COALESCE(%s, "familyHead"),
                   "agencyName" = COALESCE(%s, "agencyName"),
                   "expectedMutamers" = COALESCE(%s::jsonb, "expectedMutamers"),
                   "departureCity" = COALESCE(%s, "departureCity"),
                   "arrivalCity" = COALESCE(%s, "arrivalCity"),
                   "subAgentId" = COALESCE(%s, "subAgentId"),
                   "totalAmount" = CASE WHEN %s THEN %s ELSE "totalAmount" END,
                   "airline" = COALESCE(%s, "airline"),
                   "flightNumber" = COALESCE(%s, "flightNumber"),
                   "departureDate" = COALESCE(%s::timestamp, "departureDate"),
                   "returnDate" = COALESCE(%s::timestamp, "returnDate"),
                   "hotelName" = COALESCE(%s, "hotelName"),
                   "hotelCity" = COALESCE(%s, "hotelCity"),
                   "checkIn" = COALESCE(%s::timestamp, "checkIn"),
                   "checkOut" = COALESCE(%s::timestamp, "checkOut"),
                   "roomType" = COALESCE(%s, "roomType"),
                   "roomCount" = COALESCE(%s, "roomCount"),
                   "scanUrls" = CASE
                     WHEN %s::text IS NULL THEN "scanUrls"
                     WHEN %s = ANY("scanUrls") THEN "scanUrls"
                     ELSE array_append("scanUrls", %s)
                   END,
                   "sourceKind" = COALESCE(%s::"SourceKind", "sourceKind"),
                   "screeningStatus" = CASE
                     WHEN %s::text IS NOT NULL THEN 'PENDING'::"ScreeningStatus"
                     ELSE "screeningStatus"
                   END,
                   "updatedAt" = NOW()
                   WHERE id = %s''',
                (family_head, agency_name, expected_json, departure_city,
                 arrival_city, sub_agent_id, should_overwrite, computed_total,
                 airline, outbound_flight, outbound_date, return_date,
                 hotel_name, hotel_city, checkin_date, checkout_date,
                 room_type, room_count,
                 scan_url, scan_url, scan_url, source_kind, scan_url,
                 existing["id"]),
            )
            return {"id": existing["id"], "created": False, "totalAmount": float(computed_total or prior_total or 0)}
        # New row from agent OCR → screeningStatus PENDING so it surfaces in
        # the unified review queue. scanUrls seeded with the source media URL.
        scan_urls_seed = [scan_url] if scan_url else []
        screening = "PENDING" if scan_url else "CLEAN"
        cur.execute(
            '''INSERT INTO "Voucher"
               (id, "ubNumber", uid, status, "familyHead", "agencyName",
                "expectedMutamers", "departureCity", "arrivalCity",
                "airline", "flightNumber", "departureDate", "returnDate",
                "hotelName", "hotelCity", "checkIn", "checkOut",
                "roomType", "roomCount",
                "sourceType", "subAgentId", "totalAmount",
                "scanUrls", "sourceKind", "screeningStatus",
                "uploadSource", "createdAt", "updatedAt")
               VALUES (%s, %s, %s, 'PENDING_REVIEW', %s, %s, %s::jsonb, %s, %s,
                       %s, %s, %s::timestamp, %s::timestamp,
                       %s, %s, %s::timestamp, %s::timestamp,
                       %s, %s,
                       'SUB_AGENT', %s, %s,
                       %s, %s::"SourceKind", %s::"ScreeningStatus",
                       'WHATSAPP'::"UploadSource", NOW(), NOW())
               RETURNING id''',
            (str(uuid.uuid4()), ub_number, str(uuid.uuid4()),
             family_head, agency_name, expected_json,
             departure_city, arrival_city,
             airline, outbound_flight, outbound_date, return_date,
             hotel_name, hotel_city, checkin_date, checkout_date,
             room_type, room_count,
             sub_agent_id, computed_total or 0,
             scan_urls_seed, source_kind, screening),
        )
        return {"id": cur.fetchone()["id"], "created": True, "totalAmount": float(computed_total or 0)}


# ---------------------------------------------------------------------------
# 11. match_passport_to_voucher — silent attach without asking UB
# ---------------------------------------------------------------------------

@tool
def match_passport_to_voucher(passport_number: str, sub_agent_id: str | None = None) -> dict[str, Any]:
    """Find the voucher whose expectedMutamers list contains this passport_number.
    Optionally scoped to one sub-agent. Returns {voucher_id, ub_number} on a
    single match, {matches: [...]} when multiple, or null when none. Always
    use this BEFORE asking the user for a UB number — saves a round trip."""
    with _conn() as c, c.cursor() as cur:
        if sub_agent_id:
            cur.execute(
                '''SELECT id, "ubNumber", "familyHead" FROM "Voucher"
                   WHERE "subAgentId" = %s
                   AND "expectedMutamers" @> %s::jsonb
                   ORDER BY "createdAt" DESC LIMIT 5''',
                (sub_agent_id, json.dumps([{"passport": passport_number}])),
            )
        else:
            cur.execute(
                '''SELECT id, "ubNumber", "familyHead" FROM "Voucher"
                   WHERE "expectedMutamers" @> %s::jsonb
                   ORDER BY "createdAt" DESC LIMIT 5''',
                (json.dumps([{"passport": passport_number}]),),
            )
        rows = cur.fetchall()
    if not rows:
        return {"match": None}
    if len(rows) == 1:
        return {"match": rows[0]}
    return {"matches": rows}


@tool
def extract_pdf(message_id: str, media_idx: int = 0) -> dict[str, Any]:
    """Process a multi-page PDF attachment. Renders every page, classifies
    each (passport / voucher / id_card / payment_proof / other), and runs
    full passport OCR on each passport page.

    Returns:
        {
          "page_count": N,
          "pages": [
            {"page": 1, "kind": "passport", "mrz_status": "SUCCESS",
             "mrz": {name, passport, dob, sex, expiry, cnic_mrz},
             "viz": {place_of_birth, place_of_issue, ...}},
            {"page": 2, "kind": "voucher"},     # voucher pages flagged only
            {"page": 3, "kind": "other"}
          ]
        }

    Use this whenever classify_media returns 'pdf'. For each passport page,
    follow the normal flow: check_blacklist → match_passport_to_voucher →
    create_passport (or create_passport_draft). For voucher pages, call
    extract_voucher on the same message_id (it parses the whole PDF).
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute('SELECT media FROM "Message" WHERE id = %s', (message_id,))
        row = cur.fetchone()
        if not row or not row.get("media"):
            return {"error": "no media"}
        media = row["media"][media_idx] if isinstance(row["media"], list) else row["media"]

    try:
        with httpx.Client(timeout=120, auth=(TWILIO_SID, TWILIO_TOKEN), follow_redirects=True) as cli:
            blob = cli.get(media["url"])
            blob.raise_for_status()
            r = httpx.post(
                f"{OCR_URL}/pdf",
                files={"file": ("doc.pdf", blob.content, "application/pdf")},
                timeout=600,
            )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _capture(e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 14. finalize_passport_batch — composite tool (T2.5)
# ---------------------------------------------------------------------------
# Replaces the 17-call Phase 3 (5×check_blacklist + 5×match + 5×create +
# 1×upsert_voucher + 1×scan_drafts) with one model-side call. The composite
# orchestrates the existing tools deterministically server-side, returning
# per-passport outcomes the model can summarise in its single send_whatsapp.

@tool
def finalize_passport_batch(
    passports: list[dict],
    sub_agent_id: str | None = None,
    vouchers: list[dict] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Resolve and persist a batch of OCR'd passports against zero or more
    vouchers in one server-side orchestration. Replaces calling check_blacklist
    + match + create_passport per passport individually.

    Args:
        passports: list of dicts, each with keys:
            passport_number, surname, given_names, date_of_birth (YYYY-MM-DD),
            sex (M/F), expiry_date (YYYY-MM-DD).
            Optional: cnic, place_of_birth, booklet_number, source_message_id,
            verified (bool, default True), verification_notes.
        sub_agent_id: scope voucher matching to this sub-agent (preferred — pass
            from conversation_state.sub_agent.id).
        vouchers: optional list of voucher dicts to upsert before persisting
            passports. Each entry: ub_number (required), family_head,
            agency_name, expected_mutamers, voucher_date, package_nights.
            Routing per passport (in order):
              1. Exactly one voucher in batch -> attach to it.
              2. Voucher whose expected_mutamers contains the passport_number.
              3. match_passport_to_voucher DB lookup against open vouchers.
              4. create_passport_draft (no voucher attached).
        conversation_id: for audit logging (currently unused by this tool).

    Returns:
        {
          voucher_ids: list[str],           # all upserted voucher ids in this batch
          vouchers_processed: [             # per-voucher summary for the reply node
            {voucher_id, ub_number, passport_count}, ...
          ],
          drafts_attached: int,             # aggregated across all vouchers
          outcomes: [
            {
              passport_number,
              status: "saved" | "duplicate" | "promoted_from_draft" |
                      "blacklisted" | "cross_agent_conflict" | "saved_as_draft" |
                      "multi_match" | "error",
              voucher_id: str | None,
              voucher_ub: str | None,
              passport_id: str | None,
              flagged: bool,                # screeningStatus='FLAGGED'
              detail: dict | None,          # raw blacklist row or conflict info
              error: str | None,
            }, ...
          ],
          requires_l1: [passport_number, ...]  # passports needing escalation
        }
    """
    outcomes: list[dict] = []
    requires_l1: list[str] = []
    drafts_attached = 0

    # Step 1: upsert ALL vouchers in the burst. Build a routing index keyed by
    # passport_number (case/space-normalised) so we can attach each passport to
    # the voucher that lists it as an expected mutamer.
    upserted: list[dict[str, Any]] = []   # [{id, ub, expected_pnos: set[str]}]
    voucher_ids: list[str] = []
    voucher_passport_counts: dict[str, int] = {}
    pno_to_voucher: dict[str, dict[str, Any]] = {}

    for voucher in (vouchers or []):
        if not isinstance(voucher, dict):
            continue
        ub = (voucher.get("ub_number") or "").strip().upper()
        if not ub:
            # Skip silently — burst may include unreadable voucher pages; the
            # caller already filters None entries out.
            continue
        # Bug #11: voucher_date / package_nights aren't part of upsert_voucher's
        # signature and there's no Voucher.voucherDate / Voucher.packageNights
        # column — passing them via voucher_args would either error or silently
        # drop. Schema is canonical; remove them at the source.
        voucher_args = {
            "ub_number": ub,
            "sub_agent_id": sub_agent_id,
            "family_head": voucher.get("family_head"),
            "agency_name": voucher.get("agency_name"),
            "expected_mutamers": voucher.get("expected_mutamers") or [],
            "scan_url": voucher.get("_scan_url"),
            "source_kind": voucher.get("_source_kind"),
            "airline": voucher.get("airline"),
            "outbound_flight": voucher.get("outbound_flight"),
            "outbound_date": voucher.get("outbound_date"),
            "return_date": voucher.get("return_date"),
            "departure_city": voucher.get("departure_city"),
            "arrival_city": voucher.get("arrival_city"),
            "hotel_name": voucher.get("hotel_name"),
            "hotel_city": voucher.get("hotel_city"),
            "checkin_date": voucher.get("checkin_date"),
            "checkout_date": voucher.get("checkout_date"),
            "room_type": voucher.get("room_type"),
            "room_count": voucher.get("room_count"),
        }
        v = upsert_voucher.invoke({k: val for k, val in voucher_args.items() if val is not None})
        # Bug #2 fallout: per-voucher cross_agent_conflict must NOT abort the
        # whole batch. Skip this voucher, log via Sentry, and surface every
        # passport that would have routed to it onto requires_l1.
        if v.get("error") == "cross_agent_conflict":
            try:
                _capture(RuntimeError(
                    f"upsert_voucher cross_agent_conflict ub={ub} "
                    f"existing_agent={v.get('existing_agent')} incoming={sub_agent_id}"
                ))
            except Exception:
                pass
            for m in (voucher.get("expected_mutamers") or []):
                if not isinstance(m, dict):
                    continue
                pno_l1 = (m.get("passport") or m.get("passport_number") or "").strip().upper()
                if pno_l1 and pno_l1 not in requires_l1:
                    requires_l1.append(pno_l1)
            outcomes.append({
                "passport_number": None,
                "status": "voucher_cross_agent_conflict",
                "voucher_id": v.get("voucher_id"),
                "voucher_ub": ub,
                "passport_id": None,
                "flagged": True,
                "detail": {"existing_agent": v.get("existing_agent")},
                "error": "cross_agent_conflict",
            })
            continue
        if v.get("error"):
            return {"error": f"upsert_voucher failed for {ub}: {v['error']}"}
        vid = v.get("id")
        if not vid:
            continue

        # Index expected mutamer passport_numbers for routing.
        expected_pnos: set[str] = set()
        for m in (voucher.get("expected_mutamers") or []):
            if not isinstance(m, dict):
                continue
            mp = (m.get("passport") or m.get("passport_number") or "").strip().upper()
            if mp:
                expected_pnos.add(mp)
                if mp in pno_to_voucher:
                    # Conflict: same passport listed on two different vouchers
                    # in this burst. Don't silently route — record so we can
                    # mark the passport requires_l1 instead of guessing.
                    pno_to_voucher[mp].setdefault("conflicts", []).append({"id": vid, "ub": ub})
                else:
                    pno_to_voucher[mp] = {"id": vid, "ub": ub}

        upserted.append({"id": vid, "ub": ub, "expected_pnos": expected_pnos})
        voucher_ids.append(vid)
        voucher_passport_counts[vid] = 0

    single_voucher: dict[str, Any] | None = upserted[0] if len(upserted) == 1 else None

    # Step 2: per-passport resolve + persist. Compose existing @tools so the
    # cross-agent conflict, draft promotion, and idempotency logic stay in
    # exactly one place (create_passport).
    for p in passports:
        pno_raw = p.get("passport_number") or ""
        pno = pno_raw.strip().upper()
        if not pno:
            outcomes.append({
                "passport_number": pno_raw,
                "status": "error",
                "error": "missing passport_number",
                "voucher_id": None, "voucher_ub": None,
                "passport_id": None, "flagged": False, "detail": None,
            })
            continue

        # blacklist short-circuit
        bl = check_blacklist.invoke({"passport_number": pno})
        if bl:
            outcomes.append({
                "passport_number": pno,
                "status": "blacklisted",
                "voucher_id": None, "voucher_ub": None,
                "passport_id": None, "flagged": True,
                "detail": bl, "error": None,
            })
            requires_l1.append(pno)
            continue

        # Routing: 1) single voucher in burst, 2) expected_mutamers match,
        # 3) DB match, 4) draft fallback.
        target_vid: str | None = None
        target_ub: str | None = None

        if single_voucher is not None:
            # Bug #1: don't blindly attach every burst passport to the lone
            # voucher. If the voucher has a non-empty expected_mutamers list
            # and this pno isn't in it, treat it as a stray (e.g. someone
            # else's family member) and fall through to DB match / draft.
            # Empty expected_pnos => OCR couldn't extract the list; in that
            # case we have no way to validate, so accept the user's intent.
            sv_expected = single_voucher.get("expected_pnos") or set()
            if not sv_expected or pno in sv_expected:
                target_vid = single_voucher["id"]
                target_ub = single_voucher["ub"]
            # else: leave target_vid None so steps 3/4 below take over
        if target_vid is None and single_voucher is None:
            hit = pno_to_voucher.get(pno)
            if hit:
                # If two+ vouchers in this burst both list this passport in
                # their expected_mutamers, the right move is L1 review, not a
                # silent first-wins routing.
                if hit.get("conflicts"):
                    all_options = [{"id": hit["id"], "ub": hit["ub"]}, *hit["conflicts"]]
                    outcomes.append({
                        "passport_number": pno,
                        "status": "multi_match",
                        "voucher_id": None, "voucher_ub": None,
                        "passport_id": None, "flagged": True,
                        "detail": {"matches": all_options, "reason": "listed_on_multiple_vouchers_in_burst"},
                        "error": None,
                    })
                    requires_l1.append(pno)
                    continue
                target_vid = hit["id"]
                target_ub = hit["ub"]

        if target_vid is None:
            m = match_passport_to_voucher.invoke({
                "passport_number": pno,
                "sub_agent_id": sub_agent_id,
            })
            single = m.get("match")
            multi = m.get("matches")
            if multi:
                # Bug #9: multi_match outcomes must escalate to L1, otherwise
                # ambiguous DB matches are silently dropped from human review.
                outcomes.append({
                    "passport_number": pno,
                    "status": "multi_match",
                    "voucher_id": None, "voucher_ub": None,
                    "passport_id": None, "flagged": True,
                    "detail": {"matches": multi}, "error": None,
                })
                requires_l1.append(pno)
                continue
            if not single:
                # No voucher — save as draft instead of creating against unknown id
                draft_args = {
                    "passport_number": pno,
                    "surname": p.get("surname") or "",
                    "given_names": p.get("given_names") or "",
                    "date_of_birth": p.get("date_of_birth") or "",
                    "sex": p.get("sex") or "",
                    "expiry_date": p.get("expiry_date") or "",
                    "cnic": p.get("cnic"),
                    "place_of_birth": p.get("place_of_birth"),
                    "booklet_number": p.get("booklet_number"),
                    "source_message_id": p.get("source_message_id"),
                    "verified": p.get("verified", True),
                    "verification_notes": p.get("verification_notes"),
                }
                d = create_passport_draft.invoke({k: v for k, v in draft_args.items() if v is not None})
                outcomes.append({
                    "passport_number": pno,
                    "status": "saved_as_draft",
                    "voucher_id": None, "voucher_ub": None,
                    "passport_id": d.get("id"),
                    "flagged": d.get("screening_status") == "FLAGGED",
                    "detail": None, "error": d.get("error"),
                })
                continue
            target_vid = single["id"]
            target_ub = single.get("ubNumber")

        # create against the resolved voucher
        create_args = {
            "voucher_id": target_vid,
            "surname": p.get("surname") or "",
            "given_names": p.get("given_names") or "",
            "passport_number": pno,
            "date_of_birth": p.get("date_of_birth") or "",
            "sex": p.get("sex") or "",
            "expiry_date": p.get("expiry_date") or "",
            "cnic": p.get("cnic"),
            "place_of_birth": p.get("place_of_birth"),
            "booklet_number": p.get("booklet_number"),
            "source_message_id": p.get("source_message_id"),
            "verified": p.get("verified", True),
            "verification_notes": p.get("verification_notes"),
        }
        cp = create_passport.invoke({k: v for k, v in create_args.items() if v is not None})

        if cp.get("error"):
            outcomes.append({
                "passport_number": pno, "status": "error",
                "voucher_id": target_vid, "voucher_ub": target_ub,
                "passport_id": None, "flagged": False,
                "detail": None, "error": cp["error"],
            })
            continue
        if cp.get("conflict") == "cross_agent":
            outcomes.append({
                "passport_number": pno, "status": "cross_agent_conflict",
                "voucher_id": target_vid, "voucher_ub": target_ub,
                "passport_id": None, "flagged": True,
                "detail": cp.get("existing"), "error": None,
            })
            requires_l1.append(pno)
            continue
        if cp.get("duplicate"):
            outcomes.append({
                "passport_number": pno, "status": "duplicate",
                "voucher_id": target_vid, "voucher_ub": target_ub,
                "passport_id": cp.get("id"), "flagged": False,
                "detail": None, "error": None,
            })
            if target_vid in voucher_passport_counts:
                voucher_passport_counts[target_vid] += 1
            continue
        # promoted_from_draft surfaced via screening_status flag if present;
        # treat as 'saved' but mark detail so the reply can mention it.
        status = "saved"
        if cp.get("promoted_from_draft"):
            status = "promoted_from_draft"
        flagged = cp.get("screening_status") == "FLAGGED"
        outcomes.append({
            "passport_number": pno, "status": status,
            "voucher_id": target_vid, "voucher_ub": target_ub,
            "passport_id": cp.get("id"), "flagged": flagged,
            "detail": None, "error": None,
        })
        if target_vid in voucher_passport_counts:
            voucher_passport_counts[target_vid] += 1

    # Step 3: scan_drafts_for_voucher for every voucher we upserted.
    for u in upserted:
        try:
            sd = scan_drafts_for_voucher.invoke({"voucher_id": u["id"]})
            attached = sd.get("attached") or []
            drafts_attached += len(attached) if isinstance(attached, list) else 0
        except Exception as e:
            _capture(e)
            pass

    vouchers_processed = [
        {
            "voucher_id": u["id"],
            "ub_number": u["ub"],
            "passport_count": voucher_passport_counts.get(u["id"], 0),
        }
        for u in upserted
    ]

    return {
        "voucher_ids": voucher_ids,
        "vouchers_processed": vouchers_processed,
        "drafts_attached": drafts_attached,
        "outcomes": outcomes,
        "requires_l1": requires_l1,
    }


ALL_TOOLS = [
    conversation_state,
    classify_media,
    extract_passport,
    extract_pdf,
    extract_voucher,
    upsert_voucher,
    scan_drafts_for_voucher,
    match_passport_to_voucher,
    create_passport,
    create_passport_draft,
    finalize_passport_batch,
    lookup_agent,
    lookup_voucher,
    check_blacklist,
    send_whatsapp,
    escalate_to_l1,
]
