"""Auto-grouping (dashboard-side) test helpers.

Seeds Voucher + Passport rows, triggers `formGroupsWithStrategy('AUTO_OPTIMIZE')`,
then asserts on resulting SubmissionGroup rows.

TODO(unknown): The runner currently invokes the auto-form via
  POST {DASHBOARD_BASE_URL}/api/groups/auto-form
which requires a manager-plus session cookie. If that endpoint isn't reachable
from the test runner (mac → prod), set `GROUPING_INVOCATION = "tsx"` and we
shell out to `npx tsx --eval` against $REPO_ROOT instead. The user can pick
the cleanest path; both code paths are stubbed below.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg


GROUPING_INVOCATION = os.environ.get("GROUPING_INVOCATION", "api")  # "api" | "tsx"
DASHBOARD_BASE_URL = os.environ.get("DASHBOARD_BASE_URL", "https://passport.delveon.com")
DASHBOARD_SESSION_COOKIE = os.environ.get("DASHBOARD_SESSION_COOKIE", "")
REPO_ROOT = os.environ.get("REPO_ROOT", "/var/www/umrahflow")


@dataclass
class GroupSeed:
    voucher_count: int
    passports_per_voucher: list[int]   # parallel to voucher_count
    pool: str = "UB"                   # IataPool enum: UB | UR
    flagged_per_voucher: list[int] | None = None
    # Per-voucher totalAmount / amountReceived (parallel arrays). Default 0/0
    # which trivially satisfies the paid>=total filter. Set both to test
    # unpaid-exclusion / partial-payment / VIP-override behavior.
    total_amount_per_voucher: list[float] | None = None
    paid_per_voucher: list[float] | None = None
    # Whether to mark the seeded SubAgent as VIP (overrides the paid>=total
    # filter). Applies to the WHOLE seed batch since each scenario uses its
    # own per-run sub_agent.
    sub_agent_vip: bool = False


def seed_vouchers_and_passports(
    conn: psycopg.Connection, sub_agent_id: str, seed: GroupSeed,
) -> list[str]:
    """Insert N vouchers each with its passport count. Returns voucher IDs."""
    voucher_ids: list[str] = []
    flagged = seed.flagged_per_voucher or [0] * seed.voucher_count
    totals = seed.total_amount_per_voucher or [0.0] * seed.voucher_count
    paids = seed.paid_per_voucher or [0.0] * seed.voucher_count
    with conn.cursor() as cur:
        if seed.sub_agent_vip:
            cur.execute(
                'UPDATE "SubAgent" SET "isVip" = true WHERE id = %s',
                (sub_agent_id,),
            )
        for i in range(seed.voucher_count):
            vid = "tv_" + uuid.uuid4().hex[:16]
            ub = f"UB-T{int.from_bytes(uuid.uuid4().bytes[:3], 'big') % 1_000_000:06d}"
            cur.execute(
                '''INSERT INTO "Voucher"
                   (id, uid, "ubNumber", "subAgentId", status, pool,
                    "totalAmount", "amountReceived",
                    "createdAt", "updatedAt")
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())''',
                # loadEligibleVouchers filters on status='APPROVED' AND
                # (paid >= total OR isVip). The seeder lets each scenario
                # express that filter explicitly.
                (vid, "uid_" + vid, ub, sub_agent_id, "APPROVED", seed.pool,
                 totals[i], paids[i]),
            )
            voucher_ids.append(vid)
            n = seed.passports_per_voucher[i]
            n_flagged = flagged[i]
            for j in range(n):
                pid = "tp_" + uuid.uuid4().hex[:16]
                pno = f"T{uuid.uuid4().hex[:8].upper()}"
                screening = "FLAGGED" if j < n_flagged else "CLEAN"
                # Passport has no subAgentId/source columns — voucherId carries
                # the link to the sub-agent through the Voucher row.
                cur.execute(
                    '''INSERT INTO "Passport"
                       (id, "passportNumber", "voucherId",
                        "screeningStatus", "createdAt", "updatedAt")
                       VALUES (%s, %s, %s, %s, NOW(), NOW())''',
                    (pid, pno, vid, screening),
                )
    return voucher_ids


def cleanup_seed(conn: psycopg.Connection, voucher_ids: list[str]) -> None:
    if not voucher_ids:
        return
    with conn.cursor() as cur:
        # Order matters: read group ids BEFORE deleting Passport rows, then
        # delete Passport, then delete SubmissionGroup, then Voucher.
        cur.execute(
            '''SELECT DISTINCT "groupId" FROM "Passport"
               WHERE "voucherId" = ANY(%s::text[]) AND "groupId" IS NOT NULL''',
            (voucher_ids,),
        )
        # connect() defaults to dict_row factory.
        group_ids = [r["groupId"] for r in cur.fetchall() if r.get("groupId")]
        cur.execute(
            'DELETE FROM "Passport" WHERE "voucherId" = ANY(%s::text[])', (voucher_ids,)
        )
        if group_ids:
            cur.execute(
                'DELETE FROM "SubmissionGroup" WHERE id = ANY(%s::text[])', (group_ids,)
            )
        cur.execute(
            'DELETE FROM "Voucher" WHERE id = ANY(%s::text[])', (voucher_ids,)
        )


def trigger_auto_group() -> dict[str, Any]:
    """Kick off AUTO_OPTIMIZE grouping. Returns the response body or raises."""
    if GROUPING_INVOCATION == "tsx":
        return _trigger_via_tsx()
    return _trigger_via_api()


def _trigger_via_api() -> dict[str, Any]:
    import httpx
    if not DASHBOARD_SESSION_COOKIE:
        raise RuntimeError(
            "DASHBOARD_SESSION_COOKIE not set; required for /api/groups/auto-form. "
            "Or set GROUPING_INVOCATION=tsx."
        )
    r = httpx.post(
        f"{DASHBOARD_BASE_URL}/api/groups/auto-form",
        headers={"Cookie": DASHBOARD_SESSION_COOKIE, "Content-Type": "application/json"},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


def _trigger_via_tsx() -> dict[str, Any]:
    """Run the TS function directly via `npx tsx --eval` in REPO_ROOT.

    This is the simplest path on the prod VM where the runner has shell access
    + the source tree, but no valid web session.
    """
    # tsx --eval wraps the module so named TS exports come back under .default
    # for some shapes; cover both. Use the relative path because the @/ alias
    # isn't resolved without --tsconfig + a .ts file context.
    script = (
        "import('./src/lib/auto-group').then(async m => { "
        "const fn = (m && m.formGroupsWithStrategy) || (m.default && m.default.formGroupsWithStrategy); "
        "if (!fn) throw new Error('formGroupsWithStrategy not found; keys=' + Object.keys(m)); "
        "const r = await fn('AUTO_OPTIMIZE', {triggeredBy:'system-bot'}); "
        "console.log(JSON.stringify(r)); }).catch(e => { "
        "console.error(e && e.stack ? e.stack : e); process.exit(1); })"
    )
    proc = subprocess.run(
        ["npx", "tsx", "--eval", script],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tsx invocation failed: {proc.stderr[:500]}")
    # last JSON line is our payload
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except (TypeError, ValueError):
                continue
    return {}


def fetch_groups_for_vouchers(
    conn: psycopg.Connection, voucher_ids: list[str],
) -> list[dict[str, Any]]:
    """Return groups that contain any passport belonging to seeded vouchers."""
    if not voucher_ids:
        return []
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            '''SELECT g.id, g.code, g.pool, g.status,
                      COUNT(p.id) AS passport_count
               FROM "SubmissionGroup" g
               JOIN "Passport" p ON p."groupId" = g.id
               WHERE p."voucherId" = ANY(%s::text[])
               GROUP BY g.id''',
            (voucher_ids,),
        )
        return list(cur.fetchall())


def assert_grouping(
    groups: list[dict], expect: dict,
) -> list[str]:
    """Validate group_count, max passports per group, pool isolation."""
    failures: list[str] = []
    if "group_count" in expect and len(groups) != int(expect["group_count"]):
        failures.append(f"group_count expected {expect['group_count']} got {len(groups)}")
    cap = int(expect.get("max_per_group") or 50)
    for g in groups:
        if int(g.get("passport_count") or 0) > cap:
            failures.append(
                f"group {g.get('code')} has {g.get('passport_count')} passports (cap {cap})"
            )
    if "pools" in expect:
        seen_pools = {g.get("pool") for g in groups}
        for p in expect["pools"]:
            if p not in seen_pools:
                failures.append(f"expected pool {p} missing from groups")
    if "min_passport_total" in expect:
        total = sum(int(g.get("passport_count") or 0) for g in groups)
        if total < int(expect["min_passport_total"]):
            failures.append(
                f"grouped {total} passports, expected at least {expect['min_passport_total']}"
            )
    return failures
