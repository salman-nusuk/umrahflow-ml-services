# Agent E2E Test Suite

End-to-end harness that simulates WhatsApp users by inserting `Message` rows
directly into Postgres, lets the agent's drain loop pick them up, then asserts
on DB state + outbound replies. An LLM-as-judge grades reply tone/correctness.

## Run modes

```bash
# from agent-service/
python -m tests.agent.runner --list
python -m tests.agent.runner --scenario all
python -m tests.agent.runner --scenario single-voucher-happy-path
python -m tests.agent.runner --tag burst-routing
python -m tests.agent.runner --probe-fixtures
python -m tests.agent.runner --no-judge
```

## Where to run

Recommended: directly on the prod VM (where the agent + Postgres + pm2 live).

```bash
ssh prod
cd /var/www/umrahflow/agent-service
source venv/bin/activate
pip install -r tests/agent/requirements.txt
python -m tests.agent.runner --scenario single-voucher-happy-path
```

The runner:

1. Sets `pm2 set umrahflow-agent:TEST_MODE 1` and reloads.
2. Parks every other `botEnabled=true` conversation.
3. Creates a sandbox `Conversation` per scenario, inserts `Message` rows,
   waits for `processedAt`, asserts.
4. Cleans up: re-enables parked conversations, unsets TEST_MODE, deletes
   sandbox rows.

## Environment

- `DATABASE_URL` — required. If unset the runner reads `$REPO_ROOT/.env`.
- `REPO_ROOT` — defaults to `/var/www/umrahflow`.
- `VM_HOST` — host the agent uses to fetch fixture URLs (defaults `127.0.0.1`).
- `TEST_HTTP_PORT` — local fixture server port (default 8765).
- `ANTHROPIC_API_KEY` — needed for `judge.py`. Skip with `--no-judge`.
- `GROUPING_INVOCATION` — `api` (default) or `tsx` for grouping scenarios.
  - `api` mode also requires `DASHBOARD_BASE_URL` and `DASHBOARD_SESSION_COOKIE`.
  - `tsx` mode shells out to `npx tsx --eval` against `$REPO_ROOT`.

## Fixtures

See `fixtures/README.md`. Drop your AGOG/, BAAB E KABA/, etc. under fixtures/
(or symlink). Run `--probe-fixtures` to discover ub_number + passport_numbers
and paste into `fixtures/manifest.yml`.

## Scenarios

`scenarios.yml` is the canonical list. Each entry has `id`, optional `tags`,
optional `setup.sql` / `cleanup_sql`, an `inputs` list, and an `expect` block
(db / reply / trace / latency_ms_max).

To add a new scenario, append a YAML entry. To add a grouping scenario, set
`kind: grouping` and provide a `seed:` block instead of `inputs:`.
