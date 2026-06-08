#!/usr/bin/env bash
# test_burst.sh — exercise the agent on a synthetic 6-image burst without
# touching real Twilio or any agency phone. Run on the VM that hosts the
# agent + ocr sidecars + Postgres.
#
# Prereqs on the VM:
#   - JPEGs already uploaded to $IMG_DIR (default /tmp/test-burst/)
#   - psql installed and DATABASE_URL exported (or .env at repo root)
#   - PM2 manages umrahflow-agent
#
# Usage:
#   bash test_burst.sh                     # creates a sandbox conversation
#   bash test_burst.sh <existing_cid>      # reuses your conversation
#
# Flags via env:
#   IMG_DIR   directory of test JPEGs              (default /tmp/test-burst)
#   PORT      http.server port for static images   (default 8765)
#   WAIT_S    seconds to wait for drain to finish  (default 90)
#   REPO_ROOT path to umrahflow repo               (default /opt/umrahflow)

set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

IMG_DIR="${IMG_DIR:-/tmp/test-burst}"
PORT="${PORT:-8765}"
WAIT_S="${WAIT_S:-90}"
REPO_ROOT="${REPO_ROOT:-/var/www/umrahflow}"

# pull DATABASE_URL from repo .env if not already in env
if [[ -z "${DATABASE_URL:-}" && -f "$REPO_ROOT/.env" ]]; then
  DATABASE_URL="$(grep -E '^DATABASE_URL=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  export DATABASE_URL
fi
[[ -n "${DATABASE_URL:-}" ]] || { echo "DATABASE_URL not set"; exit 1; }

# Strip Prisma-only query params (schema, connection_limit, pgbouncer, connect_timeout)
# that psql doesn't understand. Mirrors agent.py:_psycopg_url.
PSQL_URL="$(python3 - "$DATABASE_URL" <<'PY'
import sys
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
u = urlparse(sys.argv[1])
keep = [(k, v) for k, v in parse_qsl(u.query)
        if k not in ("schema", "connection_limit", "pgbouncer", "connect_timeout")]
print(urlunparse(u._replace(query=urlencode(keep))))
PY
)"

# ---------------------------------------------------------------- helpers
psql_q() { psql "$PSQL_URL" -At -v ON_ERROR_STOP=1 -c "$1"; }
say()    { printf '\n\033[36m== %s ==\033[0m\n' "$*"; }

cleanup() {
  say "cleanup"
  if [[ -n "${HTTP_PID:-}" ]] && kill -0 "$HTTP_PID" 2>/dev/null; then
    kill "$HTTP_PID" 2>/dev/null || true
  fi
  # belt-and-braces: catch any orphaned http.server bound to $PORT
  pkill -f "http.server $PORT" 2>/dev/null || true
  echo "stopped http.server"
  if [[ -n "${PARKED_FILE:-}" && -s "$PARKED_FILE" ]]; then
    while IFS= read -r pcid; do
      [[ -z "$pcid" ]] && continue
      psql_q "UPDATE \"Conversation\" SET \"botEnabled\"=true WHERE id='$pcid';" >/dev/null
    done < "$PARKED_FILE"
    echo "re-enabled $(wc -l <"$PARKED_FILE") parked conversation(s)"
    rm -f "$PARKED_FILE"
  fi
  if [[ -n "${ENV_TOUCHED:-}" ]]; then
    # Explicit off-switch: set to 0 (truthy off) AND unset, then reload.
    # `pm2 reload --update-env` does not reliably drop unset vars from a
    # running process — must overwrite with a falsy value the code accepts.
    pm2 set umrahflow-agent:TEST_MODE 0 >/dev/null 2>&1 || true
    pm2 reload umrahflow-agent --update-env >/dev/null
    pm2 unset umrahflow-agent:TEST_MODE >/dev/null 2>&1 || true
    echo "restored agent (TEST_MODE off)"
  fi
  # Delete the sandbox conversation we created so it doesn't pile up. Skip
  # when KEEP_SANDBOX=1 (e.g. inspecting trace afterwards).
  if [[ -n "${SANDBOX_OWNED:-}" && -n "${CID:-}" && -z "${KEEP_SANDBOX:-}" ]]; then
    psql_q "DELETE FROM \"Message\" WHERE \"conversationId\"='$CID';" >/dev/null
    psql_q "DELETE FROM \"AgentTrace\" WHERE \"conversationId\"='$CID';" >/dev/null
    for tbl in checkpoints checkpoint_writes checkpoint_blobs; do
      psql_q "DELETE FROM $tbl WHERE thread_id='$CID';" >/dev/null 2>&1 || true
    done
    psql_q "DELETE FROM \"Conversation\" WHERE id='$CID';" >/dev/null
    echo "deleted sandbox conversation $CID"
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------- prechecks
say "prechecks"
[[ -d "$IMG_DIR" ]] || { echo "missing $IMG_DIR"; exit 1; }
mapfile -t FILES < <(cd "$IMG_DIR" && ls -1 *.jpeg *.jpg 2>/dev/null || true)
[[ ${#FILES[@]} -ge 1 ]] || { echo "no jpegs in $IMG_DIR"; exit 1; }
echo "found ${#FILES[@]} image(s) in $IMG_DIR"
command -v pm2  >/dev/null || { echo "pm2 not on PATH"; exit 1; }
command -v psql >/dev/null || { echo "psql not on PATH"; exit 1; }

# ---------------------------------------------------------------- conversation
CID="${1:-}"
SANDBOX_OWNED=""
if [[ -z "$CID" ]]; then
  say "creating sandbox conversation"
  # psql -At still emits "INSERT 0 1" after a RETURNING row; capture the
  # whole output and take only the first line. Avoid `| head -n1` because
  # set -o pipefail + SIGPIPE on psql kills the script.
  _ins=$(psql_q "INSERT INTO \"Conversation\" (id, phone, \"botEnabled\", \"createdAt\", \"lastMessageAt\")
                 VALUES (gen_random_uuid()::text, '+99999$(date +%s | tail -c 7)', true, NOW(), NOW())
                 RETURNING id;")
  CID="${_ins%%$'\n'*}"
  SANDBOX_OWNED=1
  echo "CID=$CID (will be deleted on exit; set KEEP_SANDBOX=1 to retain)"
else
  echo "using existing CID=$CID (will NOT be deleted)"
  psql_q "UPDATE \"Conversation\" SET \"botEnabled\"=true WHERE id='$CID';" >/dev/null
fi

# ---------------------------------------------------------------- park others
say "parking other bot-enabled conversations"
PARKED_FILE="$(mktemp)"
psql_q "SELECT id FROM \"Conversation\" WHERE \"botEnabled\"=true AND id <> '$CID';" > "$PARKED_FILE"
parked=$(wc -l <"$PARKED_FILE")
echo "parked $parked conversation(s) (will restore on exit)"
psql_q "UPDATE \"Conversation\" SET \"botEnabled\"=false WHERE \"botEnabled\"=true AND id <> '$CID';" >/dev/null

# ---------------------------------------------------------------- wipe checkpoint + clear pending msgs
say "clearing prior checkpoint + unprocessed inbound on $CID"
for tbl in checkpoints checkpoint_writes checkpoint_blobs; do
  psql_q "DELETE FROM $tbl WHERE thread_id='$CID';" >/dev/null 2>&1 || true
done
psql_q "DELETE FROM \"Message\" WHERE \"conversationId\"='$CID' AND direction='IN' AND \"processedAt\" IS NULL;" >/dev/null
psql_q "DELETE FROM \"AgentTrace\" WHERE \"conversationId\"='$CID' AND \"startedAt\" > NOW() - interval '1 hour';" >/dev/null

# ---------------------------------------------------------------- http server
say "starting http.server on :$PORT"
# free the port if a previous run left a stray process
pkill -f "http.server $PORT" 2>/dev/null || true
sleep 0.5
# use `exec` so $! is the python3 pid, not the subshell pid
( cd "$IMG_DIR" && exec python3 -m http.server "$PORT" >/tmp/test-burst-http.log 2>&1 ) &
HTTP_PID=$!
sleep 1
kill -0 "$HTTP_PID" 2>/dev/null || { echo "http.server failed to start, see /tmp/test-burst-http.log"; exit 1; }
echo "http.server pid=$HTTP_PID"

# ---------------------------------------------------------------- insert burst
say "inserting ${#FILES[@]} inbound message(s) on $CID"
for f in "${FILES[@]}"; do
  url="http://127.0.0.1:$PORT/$f"
  # CID is uuid-shape, URL we generated ourselves and contains no single
  # quotes, so plain SQL single-quoting is safe.
  psql "$PSQL_URL" -At -v ON_ERROR_STOP=1 <<SQL >/dev/null
INSERT INTO "Message" (id, "conversationId", direction, body, media, status, "createdAt")
VALUES (gen_random_uuid()::text, '$CID', 'IN', '',
        jsonb_build_array(jsonb_build_object('url', '$url', 'contentType', 'image/jpeg')),
        'RECEIVED', NOW());
SQL
done
echo "inserted; processedAt=NULL on all rows"

# ---------------------------------------------------------------- agent in TEST_MODE
say "restarting agent with TEST_MODE=1"
pm2 set umrahflow-agent:TEST_MODE 1 >/dev/null
pm2 reload umrahflow-agent --update-env >/dev/null
ENV_TOUCHED=1
echo "agent restarted; drain will pick up the burst after the 3s debounce"

# ---------------------------------------------------------------- wait
say "waiting up to ${WAIT_S}s for the trace to land"
deadline=$(( $(date +%s) + WAIT_S ))
trace_id=""
while (( $(date +%s) < deadline )); do
  _t=$(psql_q "SELECT id FROM \"AgentTrace\" WHERE \"conversationId\"='$CID' ORDER BY \"startedAt\" DESC LIMIT 1;" 2>/dev/null || true)
  trace_id="${_t%%$'\n'*}"
  [[ -n "$trace_id" ]] && break
  printf '.'
  sleep 2
done
echo

if [[ -z "$trace_id" ]]; then
  echo "no AgentTrace landed within ${WAIT_S}s — check pm2 logs umrahflow-agent"
  exit 2
fi

# ---------------------------------------------------------------- report
say "RESULT"
psql "$PSQL_URL" <<SQL
\pset border 2
\pset format aligned

SELECT "durationMs" AS duration_ms,
       jsonb_array_length("toolCalls") AS total_calls,
       ("toolCalls" -> -1 -> 'usage') AS token_usage
FROM "AgentTrace" WHERE id='$trace_id';

\echo
\echo --- tool call breakdown ---
SELECT tc->>'name' AS tool, COUNT(*) AS n
FROM "AgentTrace", jsonb_array_elements("toolCalls") tc
WHERE id='$trace_id' AND tc->>'name' <> '_meta'
GROUP BY 1 ORDER BY 2 DESC, 1;

\echo
\echo --- final reply text ---
SELECT "finalText" FROM "AgentTrace" WHERE id='$trace_id';

\echo
\echo --- outbound rows produced (all should have TEST- sids in TEST_MODE) ---
SELECT id, status, "twilioSid", LEFT(body, 100) AS body_preview, "createdAt"
FROM "Message"
WHERE "conversationId"='$CID' AND direction='OUT'
ORDER BY "createdAt" ASC;
SQL

echo
echo "CID=$CID  trace_id=$trace_id"
echo "(cleanup runs automatically on exit: stops http.server, restores parked conversations, turns TEST_MODE off, restarts agent)"
