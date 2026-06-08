#!/usr/bin/env bash
# Nightly Postgres dump → encrypted upload to GCS.
#
# Run via crontab on the prod VM:
#     0 2 * * *  /var/www/umrahflow/scripts/backup.sh >> /var/log/umrahflow-backup.log 2>&1
#
# Required env (sourced from /var/www/umrahflow/.env):
#   DATABASE_URL                Prisma DSN; the script strips ?schema= before psql/pg_dump.
#   BACKUP_GCS_BUCKET           e.g. gs://umrahflow-backups
#   BACKUP_PASSPHRASE           OpenSSL symmetric key; rotate yearly. Min 32 chars.
#
# Optional:
#   BACKUP_RETENTION_DAYS       default 30 — local cleanup window
#   GCLOUD_SERVICE_KEY          path to a service-account JSON; falls back to
#                               the VM's default credentials if unset.

set -euo pipefail
export LC_ALL=C.UTF-8 LANG=C.UTF-8

REPO_ROOT="${REPO_ROOT:-/var/www/umrahflow}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC2046
  export $(grep -E '^(DATABASE_URL|BACKUP_GCS_BUCKET|BACKUP_PASSPHRASE|BACKUP_RETENTION_DAYS|GCLOUD_SERVICE_KEY)=' "$ENV_FILE" | xargs -I{} echo {})
fi

: "${DATABASE_URL:?DATABASE_URL must be set}"
: "${BACKUP_GCS_BUCKET:?BACKUP_GCS_BUCKET must be set (e.g. gs://umrahflow-backups)}"
: "${BACKUP_PASSPHRASE:?BACKUP_PASSPHRASE must be set (min 32 chars)}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Strip the ?schema= query param — pg_dump doesn't accept it.
CLEAN_URL="${DATABASE_URL%%\?*}"

LOCAL_DIR="${LOCAL_DIR:-/var/backups/umrahflow}"
mkdir -p "$LOCAL_DIR"

TS=$(date -u +%Y%m%dT%H%M%SZ)
BASENAME="umrahflow-$TS.sql.gz.enc"
LOCAL_PATH="$LOCAL_DIR/$BASENAME"

echo "[backup $TS] starting pg_dump"
pg_dump --no-owner --no-acl --format=plain "$CLEAN_URL" \
  | gzip -9 \
  | openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:BACKUP_PASSPHRASE \
  > "$LOCAL_PATH"

SIZE=$(stat -c %s "$LOCAL_PATH" 2>/dev/null || stat -f %z "$LOCAL_PATH")
echo "[backup $TS] dump complete: $LOCAL_PATH ($SIZE bytes)"

if [[ -n "${GCLOUD_SERVICE_KEY:-}" ]]; then
  gcloud auth activate-service-account --key-file="$GCLOUD_SERVICE_KEY" >/dev/null
fi

echo "[backup $TS] uploading to $BACKUP_GCS_BUCKET"
gsutil cp "$LOCAL_PATH" "$BACKUP_GCS_BUCKET/$BASENAME"

echo "[backup $TS] pruning local backups older than ${RETENTION_DAYS}d"
find "$LOCAL_DIR" -type f -name 'umrahflow-*.sql.gz.enc' -mtime +"$RETENTION_DAYS" -delete

echo "[backup $TS] done"
