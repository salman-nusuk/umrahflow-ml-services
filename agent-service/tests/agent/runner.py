"""End-to-end agent test runner.

Designed to run ON the prod VM (recommended — direct DB + pm2 access). Can also
run locally if DATABASE_URL + a tunnel/SSH are set up; but the simplest and
most reliable path is `ssh prod` + `python -m tests.agent.runner --scenario all`.

TODO(unknown): The grouping scenarios need an invocation path for
`formGroupsWithStrategy`. We default to POST /api/groups/auto-form with a
session cookie; set GROUPING_INVOCATION=tsx to instead shell out to
`npx tsx --eval` against $REPO_ROOT. See grouping.py.

Usage:
  python -m tests.agent.runner --list
  python -m tests.agent.runner --scenario all
  python -m tests.agent.runner --scenario single-voucher-happy-path
  python -m tests.agent.runner --tag burst-routing
  python -m tests.agent.runner --probe-fixtures
  python -m tests.agent.runner --no-judge
"""
from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg
import psycopg.rows
import yaml

# Prefer relative imports when invoked as `python -m tests.agent.runner`.
try:
    from . import assertions, grouping, judge, report
except ImportError:  # pragma: no cover - direct script run fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.agent import assertions, grouping, judge, report  # type: ignore


HERE = Path(__file__).resolve().parent
SCENARIOS_PATH = HERE / "scenarios.yml"
FIXTURES_DIR = HERE / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.yml"

DEFAULT_HTTP_PORT = int(os.environ.get("TEST_HTTP_PORT", "8765"))
DEFAULT_VM_HOST = os.environ.get("VM_HOST", "127.0.0.1")
TEST_AGENT_A = "__test_agent_A__"
TEST_AGENT_B = "__test_agent_B__"


# ----------------------------------------------------------------- DB helpers

def _psycopg_url(prisma_url: str) -> str:
    p = urlparse(prisma_url)
    keep = [
        (k, v)
        for k, v in parse_qsl(p.query)
        if k not in ("schema", "connection_limit", "pgbouncer", "connect_timeout")
    ]
    return urlunparse(p._replace(query=urlencode(keep)))


def _load_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return _psycopg_url(url)
    # Try to read from $REPO_ROOT/.env (mirrors test_burst.sh).
    repo_root = os.environ.get("REPO_ROOT", "/var/www/umrahflow")
    env_path = Path(repo_root) / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return _psycopg_url(v)
    raise RuntimeError("DATABASE_URL not set and not found in $REPO_ROOT/.env")


def connect() -> psycopg.Connection:
    return psycopg.connect(
        _load_database_url(), autocommit=True, row_factory=psycopg.rows.dict_row
    )


# ----------------------------------------------------------------- pm2 / TEST_MODE

def _pm2(*args: str) -> None:
    subprocess.run(["pm2", *args], check=False, capture_output=True)


def set_test_mode(on: bool) -> None:
    if on:
        _pm2("set", "umrahflow-agent:TEST_MODE", "1")
    else:
        _pm2("set", "umrahflow-agent:TEST_MODE", "0")
    _pm2("reload", "umrahflow-agent", "--update-env")
    if not on:
        _pm2("unset", "umrahflow-agent:TEST_MODE")


# ----------------------------------------------------------------- HTTP fixture server

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_a, **_k):  # noqa: D401 - silence access logs
        return


def start_fixture_server(serve_root: Path, port: int) -> tuple[Any, threading.Thread]:
    handler = type(
        "H", (_QuietHandler,),
        {"directory": str(serve_root)},
    )

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

        def finish_request(self, request, client_address):
            self.RequestHandlerClass(request, client_address, self, directory=str(serve_root))  # type: ignore[arg-type]

    httpd = _Server(("0.0.0.0", port), _QuietHandler)
    httpd.RequestHandlerClass.directory = str(serve_root)  # type: ignore[attr-defined]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def stop_fixture_server(httpd: Any) -> None:
    try:
        httpd.shutdown()
        httpd.server_close()
    except Exception:
        pass


# ----------------------------------------------------------------- sandbox

@dataclass
class Sandbox:
    cid: str
    sub_agent_id: str
    parked_cids: list[str]


def _ensure_test_sub_agent(conn: psycopg.Connection, name: str) -> str:
    with conn.cursor() as cur:
        cur.execute('SELECT id FROM "SubAgent" WHERE name=%s', (name,))
        row = cur.fetchone()
        if row:
            return row["id"]
        sid = "tsa_" + uuid.uuid4().hex[:16]
        cur.execute(
            '''INSERT INTO "SubAgent"
               (id, name, "contactName", phone, city, status, "createdAt")
               VALUES (%s, %s, %s, %s, %s, %s, NOW())''',
            (sid, name, "Test Contact", "+92-test-" + sid[-6:], "Test City", "ACTIVE"),
        )
        return sid


def acquire_sandbox(
    conn: psycopg.Connection, sub_agent_name: str = TEST_AGENT_A,
) -> Sandbox:
    # Always create a fresh SubAgent per scenario so cross-scenario state
    # (Vouchers/Passports under the same sub_agent_id) can't leak. The base
    # name is preserved for debug visibility; uniqueness comes from a uuid
    # suffix. Cleanup happens in release_sandbox.
    unique_name = f"{sub_agent_name}_{uuid.uuid4().hex[:8]}"
    sub_agent_id = _ensure_test_sub_agent(conn, unique_name)
    cid = uuid.uuid4().hex
    phone = f"+9999{int(time.time()) % 10_000_000}"
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO "Conversation"
               (id, phone, "subAgentId", "botEnabled", "createdAt", "lastMessageAt")
               VALUES (%s, %s, %s, true, NOW(), NOW())''',
            (cid, phone, sub_agent_id),
        )
        cur.execute(
            'SELECT id FROM "Conversation" WHERE "botEnabled"=true AND id <> %s',
            (cid,),
        )
        parked = [r["id"] for r in cur.fetchall()]
        if parked:
            cur.execute(
                'UPDATE "Conversation" SET "botEnabled"=false WHERE id = ANY(%s::text[])',
                (parked,),
            )
    return Sandbox(cid=cid, sub_agent_id=sub_agent_id, parked_cids=parked)


def release_sandbox(conn: psycopg.Connection, sb: Sandbox) -> None:
    with conn.cursor() as cur:
        for tbl in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
            try:
                cur.execute(f'DELETE FROM {tbl} WHERE thread_id=%s', (sb.cid,))
            except Exception:
                pass
        cur.execute('DELETE FROM "Message" WHERE "conversationId"=%s', (sb.cid,))
        cur.execute('DELETE FROM "AgentTrace" WHERE "conversationId"=%s', (sb.cid,))
        cur.execute('DELETE FROM "Conversation" WHERE id=%s', (sb.cid,))
        # Tear down SubAgent + its Vouchers/Passports so the next scenario
        # starts clean. Test sub_agents are uniquely created per scenario.
        # Passports are linked via Voucher.subAgentId, plus drafts (voucherId
        # IS NULL) created in this conversation window.
        cur.execute(
            '''DELETE FROM "Passport"
               WHERE "voucherId" IN (SELECT id FROM "Voucher" WHERE "subAgentId"=%s)''',
            (sb.sub_agent_id,),
        )
        cur.execute('DELETE FROM "Voucher" WHERE "subAgentId"=%s', (sb.sub_agent_id,))
        cur.execute('DELETE FROM "SubAgent" WHERE id=%s', (sb.sub_agent_id,))
        if sb.parked_cids:
            cur.execute(
                'UPDATE "Conversation" SET "botEnabled"=true WHERE id = ANY(%s::text[])',
                (sb.parked_cids,),
            )


# ----------------------------------------------------------------- message injection

def insert_inbound_text(conn: psycopg.Connection, cid: str, text: str) -> str:
    mid = uuid.uuid4().hex
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO "Message"
               (id, "conversationId", direction, body, status, "createdAt")
               VALUES (%s, %s, 'IN', %s, 'RECEIVED', NOW())''',
            (mid, cid, text),
        )
    return mid


def insert_inbound_media(
    conn: psycopg.Connection, cid: str, urls_with_types: list[tuple[str, str]],
    body: str = "",
) -> str:
    mid = uuid.uuid4().hex
    media_json = [{"url": u, "contentType": ct} for u, ct in urls_with_types]
    with conn.cursor() as cur:
        cur.execute(
            '''INSERT INTO "Message"
               (id, "conversationId", direction, body, media, status, "createdAt")
               VALUES (%s, %s, 'IN', %s, %s::jsonb, 'RECEIVED', NOW())''',
            (mid, cid, body, _json_dumps(media_json)),
        )
    return mid


def _json_dumps(obj: Any) -> str:
    import json as _json
    return _json.dumps(obj)


# ----------------------------------------------------------------- waiting

def wait_for_processed(
    conn: psycopg.Connection, cid: str, message_ids: list[str], timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT COUNT(*) AS n FROM "Message" '
                'WHERE id = ANY(%s::text[]) AND "processedAt" IS NOT NULL',
                (message_ids,),
            )
            n = int((cur.fetchone() or {}).get("n") or 0)
        if n >= len(message_ids):
            return True
        time.sleep(2.0)
    return False


# ----------------------------------------------------------------- scenarios

def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        raise RuntimeError(f"missing {SCENARIOS_PATH}")
    data = yaml.safe_load(SCENARIOS_PATH.read_text())
    if not isinstance(data, list):
        raise RuntimeError("scenarios.yml must be a top-level list")
    return data


def filter_scenarios(scenarios: list[dict], spec: str | None, tag: str | None) -> list[dict]:
    if spec and spec != "all":
        return [s for s in scenarios if s.get("id") == spec]
    if tag:
        return [s for s in scenarios if tag in (s.get("tags") or [])]
    return scenarios


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return yaml.safe_load(MANIFEST_PATH.read_text()) or {}


# ----------------------------------------------------------------- single scenario

def _stage_name(rel: str, src: Path) -> str:
    """Flatten an `agent_A/voucher.pdf`-style relative path into a unique
    filename so the staging http.server can serve every fixture even when
    sibling folders share a basename like `voucher.pdf`. Suffix preserved
    so content-type detection still works."""
    safe = rel.replace("/", "__").replace(" ", "_")
    return safe


def _resolve_media_urls(media_paths: list[str], port: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for rel in media_paths:
        # relative to fixtures/, resolved through manifest source paths if needed
        path = Path(rel)
        if not path.is_absolute():
            path = FIXTURES_DIR / rel
        if not path.exists():
            raise FileNotFoundError(f"fixture missing: {path}")
        ct, _ = mimetypes.guess_type(str(path))
        url_name = _stage_name(rel, path)
        out.append((f"http://{DEFAULT_VM_HOST}:{port}/{url_name}", ct or "application/octet-stream"))
    return out


def _stage_fixtures(media_paths: list[str], stage: Path) -> None:
    """Copy referenced fixture files into a flat staging dir for http.server.
    Names are flattened (subdir__file.ext) so duplicates across folders
    don't overwrite each other."""
    import shutil
    stage.mkdir(parents=True, exist_ok=True)
    for rel in media_paths:
        src = Path(rel) if Path(rel).is_absolute() else (FIXTURES_DIR / rel)
        if not src.exists():
            raise FileNotFoundError(f"fixture missing: {src}")
        dst = stage / _stage_name(rel, src)
        # Always overwrite to pick up edits between runs.
        shutil.copy2(src, dst)


class _SetupFailed(Exception):
    """Sentinel — raised after setup-SQL failure to skip assertions but still hit finally."""


def run_scenario(
    scenario: dict, *, run_judge: bool = True, http_port: int = DEFAULT_HTTP_PORT,
) -> report.ScenarioReport:
    sid = scenario.get("id") or "<unnamed>"
    t0 = time.monotonic()
    cost = 0.0
    failures: list[str] = []

    # Grouping scenarios run a different code path entirely.
    if (scenario.get("kind") or "").lower() == "grouping":
        return _run_grouping_scenario(scenario, t0)

    sub_agent_name = scenario.get("sub_agent") or TEST_AGENT_A
    inputs = scenario.get("inputs") or []
    expect = scenario.get("expect") or {}
    latency_cap = int(expect.get("latency_ms_max") or 90_000)

    # collect fixture paths once for staging
    all_media: list[str] = []
    for inp in inputs:
        for m in inp.get("media") or []:
            all_media.append(m)

    httpd = None
    stage_dir = HERE / ".http_stage"
    if all_media:
        try:
            _stage_fixtures(all_media, stage_dir)
        except FileNotFoundError as exc:
            return report.ScenarioReport(
                id=sid, passed=False, duration_s=time.monotonic() - t0,
                failures=[f"fixture missing: {exc}"],
            )
        try:
            httpd, _ = start_fixture_server(stage_dir, http_port)
        except OSError as exc:
            return report.ScenarioReport(
                id=sid, passed=False, duration_s=time.monotonic() - t0,
                failures=[f"http.server failed on :{http_port}: {exc}"],
            )

    conn = connect()
    sb = acquire_sandbox(conn, sub_agent_name=sub_agent_name)
    sandbox_start_ms = int(time.time() * 1000)
    inserted_ids: list[str] = []

    cleanup_sql = ((scenario.get("setup") or {}).get("cleanup_sql") or "").strip()

    setup_failed = False
    try:
        setup_sql = ((scenario.get("setup") or {}).get("sql") or "").strip()
        if setup_sql:
            try:
                with conn.cursor() as cur:
                    cur.execute(setup_sql)
            except Exception as exc:
                # Don't kill the batch — surface as a scenario failure so
                # subsequent scenarios still run. Skip the input loop and
                # assertions, but still run the finally cleanup below.
                failures.append(f"[setup] {type(exc).__name__}: {exc}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                setup_failed = True

        if setup_failed:
            inputs = []

        for inp in inputs:
            if "delay_ms" in inp:
                time.sleep(float(inp["delay_ms"]) / 1000.0)
                continue
            text = inp.get("text") or ""
            media = inp.get("media") or []
            if media:
                # WhatsApp delivers each media item as a SEPARATE webhook →
                # separate Message row. The agent's classify pass walks per-
                # message, so dumping N media into one row would only ever
                # classify the first. Mirror real delivery: one row per file,
                # text on the first row only.
                resolved = _resolve_media_urls(media, http_port)
                for i, (url, ct) in enumerate(resolved):
                    mid = insert_inbound_media(
                        conn, sb.cid, [(url, ct)], body=(text if i == 0 else ""),
                    )
                    inserted_ids.append(mid)
            else:
                mid = insert_inbound_text(conn, sb.cid, text)
                inserted_ids.append(mid)

        if setup_failed:
            raise _SetupFailed()

        ok = wait_for_processed(conn, sb.cid, inserted_ids, (latency_cap + 30_000) / 1000.0)
        if not ok:
            failures.append(
                f"timed out waiting for processedAt on {len(inserted_ids)} inbound row(s)"
            )

        # collect results
        outbound = assertions.fetch_outbound(conn, sb.cid)
        trace = assertions.fetch_latest_trace(conn, sb.cid)

        for chk in (
            assertions.check_db(
                conn, sb.sub_agent_id, expect.get("db") or {},
                since_ms=sandbox_start_ms,
            ),
            assertions.check_reply(outbound, expect.get("reply") or {}),
            assertions.check_trace(trace, expect.get("trace") or {}),
            assertions.check_latency(
                int((trace or {}).get("durationMs") or 0) if trace else None,
                expect,
            ),
        ):
            failures.extend(f"[{chk.name}] {f}" for f in chk.failures)

        # Judge step
        tone_criterion = (expect.get("reply") or {}).get("tone_judge")
        if run_judge and tone_criterion and outbound:
            reply_text = "\n---\n".join((m.get("body") or "") for m in outbound)
            try:
                jr = judge.grade(tone_criterion, reply_text)
                cost += jr.cost_usd
                if not jr.passed:
                    failures.append(f"[judge] {jr.reason} (score={jr.score:.2f})")
            except Exception as exc:
                failures.append(f"[judge] grading failed: {exc}")

    except _SetupFailed:
        pass
    finally:
        if cleanup_sql:
            try:
                with conn.cursor() as cur:
                    cur.execute(cleanup_sql)
            except Exception as exc:
                failures.append(f"[cleanup] {exc}")
        try:
            if not os.environ.get("KEEP_SANDBOX"):
                release_sandbox(conn, sb)
            else:
                print(f"[runner] KEEP_SANDBOX=1 — leaving cid={sb.cid} sub_agent_id={sb.sub_agent_id}")
        finally:
            conn.close()
            if httpd is not None:
                stop_fixture_server(httpd)

    return report.ScenarioReport(
        id=sid, passed=not failures, duration_s=time.monotonic() - t0,
        cost_usd=cost, failures=failures,
    )


def _run_grouping_scenario(scenario: dict, t0: float) -> report.ScenarioReport:
    sid = scenario.get("id") or "<unnamed>"
    expect = scenario.get("expect") or {}
    seed_cfg = scenario.get("seed") or {}
    failures: list[str] = []
    conn = connect()
    voucher_ids: list[str] = []
    try:
        sub_agent_id = _ensure_test_sub_agent(conn, TEST_AGENT_A)
        seed = grouping.GroupSeed(
            voucher_count=int(seed_cfg.get("voucher_count") or 0),
            passports_per_voucher=list(seed_cfg.get("passports_per_voucher") or []),
            pool=str(seed_cfg.get("pool") or "FOC"),
            flagged_per_voucher=seed_cfg.get("flagged_per_voucher"),
            total_amount_per_voucher=seed_cfg.get("total_amount_per_voucher"),
            paid_per_voucher=seed_cfg.get("paid_per_voucher"),
            sub_agent_vip=bool(seed_cfg.get("sub_agent_vip") or False),
        )
        voucher_ids = grouping.seed_vouchers_and_passports(conn, sub_agent_id, seed)
        try:
            grouping.trigger_auto_group()
        except Exception as exc:
            failures.append(f"[group] trigger failed: {exc}")
        groups = grouping.fetch_groups_for_vouchers(conn, voucher_ids)
        failures.extend(f"[group] {f}" for f in grouping.assert_grouping(groups, expect))
    finally:
        try:
            grouping.cleanup_seed(conn, voucher_ids)
        finally:
            conn.close()
    return report.ScenarioReport(
        id=sid, passed=not failures, duration_s=time.monotonic() - t0,
        failures=failures,
    )


# ----------------------------------------------------------------- probe

def probe_fixtures() -> int:
    """OCR each manifest fixture and print findings so the user can paste
    expected ub_number / passport_numbers into manifest.yml."""
    manifest = load_manifest()
    if not manifest:
        print("no fixtures/manifest.yml — see fixtures/README.md for format")
        return 1
    # We delegate OCR to the agent's own tools (extract_voucher / extract_passport)
    # by importing them lazily. Importing tools.py requires DATABASE_URL.
    try:
        sys.path.insert(0, str(HERE.parents[1]))  # agent-service/
        from tools import extract_passport, extract_voucher  # type: ignore
    except Exception as exc:
        print(f"could not import agent tools (need DATABASE_URL set): {exc}")
        return 2
    print(f"probing {len(manifest)} fixture group(s)...")
    for name, info in manifest.items():
        src = Path(info.get("source_path") or "")
        print(f"\n=== {name} :: {src} ===")
        if not src.exists():
            print(f"  (missing) {src}")
            continue
        for f in sorted(src.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".pdf"):
                continue
            print(f"  {f.name} -> probe (OCR via agent.tools requires a real "
                  f"message_id; skipping in probe-mode placeholder)")
    print("\nNote: full OCR probing requires inserting a Message row with the URL"
          " then calling extract_*. Run a single-shot scenario instead and read"
          " the trace if you need raw OCR output.")
    return 0


# ----------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tests.agent.runner")
    p.add_argument("--scenario", default=None, help="scenario id or 'all'")
    p.add_argument("--tag", default=None, help="filter by tag")
    p.add_argument("--list", action="store_true", help="print scenarios and exit")
    p.add_argument("--no-judge", action="store_true", help="skip LLM tone grading")
    p.add_argument("--probe-fixtures", action="store_true")
    p.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    p.add_argument("--keep-going", action="store_true", help="continue on failures")
    args = p.parse_args(argv)

    if args.probe_fixtures:
        return probe_fixtures()

    scenarios = load_scenarios()

    if args.list:
        for s in scenarios:
            tags = ",".join(s.get("tags") or [])
            print(f"{s.get('id'):45s}  [{tags}]  {s.get('description','')}")
        return 0

    selected = filter_scenarios(scenarios, args.scenario, args.tag)
    if not selected:
        print(f"no scenarios matched (--scenario={args.scenario} --tag={args.tag})")
        return 1

    # Suite-level setup: TEST_MODE on. Cleanup on every exit path.
    set_test_mode(True)
    reports: list[report.ScenarioReport] = []
    try:
        report.print_header(len(selected))
        for i, sc in enumerate(selected, 1):
            r = run_scenario(sc, run_judge=not args.no_judge, http_port=args.http_port)
            reports.append(r)
            report.print_scenario(i, len(selected), r)
            if not r.passed and not args.keep_going and len(selected) == 1:
                break
        report.print_summary(reports)
    finally:
        set_test_mode(False)

    return 0 if all(r.passed or r.skipped for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
