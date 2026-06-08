"""Build a ground-truth OCR reference using Sonnet 4.6 over a sample of real
Pakistani-passport scans.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... \
    python scripts/build_reference.py \
        --src '/Users/afaqahmad/Downloads/ses 11111' \
        --out tests/fixtures/ground_truth.json \
        --sample 256 \
        --concurrency 8
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

PROMPT = """You are extracting fields from a Pakistani passport bio-page scan to
build a reference dataset. Read carefully — null is better than wrong.

Return ONLY this JSON (no markdown, no commentary):
{
  "mrz": {
    "name": null,
    "passport": null,
    "dob": null,
    "sex": null,
    "expiry": null,
    "cnic_mrz": null
  },
  "viz": {
    "place_of_birth": null,
    "place_of_issue": null,
    "date_of_issue": null,
    "father_name": null,
    "husband_name": null,
    "booklet_number": null,
    "cnic": null,
    "tracking_number": null
  }
}

RULES:
- mrz.name: "SURNAME GIVENNAMES" exactly as in the MRZ band (the two lines of \
chevron-separated text at the bottom). Uppercase, single space between surname \
and given names.
- mrz.passport: 6-12 alphanumeric characters from MRZ position 1 of line 2.
- mrz.dob, mrz.expiry: format YYYY-MM-DD. The MRZ encodes them as YYMMDD; \
expand the year using the convention DOB year >= current_year-100.
- mrz.sex: "M" or "F".
- mrz.cnic_mrz: any 13-digit number printed in the MRZ optional-data field. Null \
if not present or shorter.
- viz.place_of_birth: city only, e.g. "GUJRAT" not "GUJRAT, PAK".
- viz.place_of_issue: literal text. May be "PAKISTAN" or a city name. NEVER \
output a country code like "PAK".
- viz.date_of_issue: YYYY-MM-DD.
- viz.father_name: only if printed; for adult women a husband's name may appear \
instead. Output exactly what is printed.
- viz.husband_name: same — only if printed.
- viz.booklet_number: serial in the top-right corner starting with a letter \
(e.g. M9161105). NOT the same as the passport number.
- viz.cnic: 13-digit Pakistani national ID, formatted XXXXX-XXXXXXX-X. Null if \
not exactly 13 digits visible.
- viz.tracking_number: 11-digit number labeled "Tracking Number" or "Citizenship \
Number". Distinct from CNIC.

If the image is not a Pakistani passport bio-page, return all nulls."""


def encode_image(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix.lstrip("."), "image/jpeg"
    )
    return base64.b64encode(raw).decode(), media


EXTRACTION_TOOL = {
    "name": "record_passport_extraction",
    "description": "Record the parsed fields from one Pakistani passport.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mrz": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "passport": {"type": ["string", "null"]},
                    "dob": {"type": ["string", "null"]},
                    "sex": {"type": ["string", "null"]},
                    "expiry": {"type": ["string", "null"]},
                    "cnic_mrz": {"type": ["string", "null"]},
                },
                "required": ["name", "passport", "dob", "sex", "expiry", "cnic_mrz"],
            },
            "viz": {
                "type": "object",
                "properties": {
                    "place_of_birth": {"type": ["string", "null"]},
                    "place_of_issue": {"type": ["string", "null"]},
                    "date_of_issue": {"type": ["string", "null"]},
                    "father_name": {"type": ["string", "null"]},
                    "husband_name": {"type": ["string", "null"]},
                    "booklet_number": {"type": ["string", "null"]},
                    "cnic": {"type": ["string", "null"]},
                    "tracking_number": {"type": ["string", "null"]},
                },
                "required": [
                    "place_of_birth", "place_of_issue", "date_of_issue",
                    "father_name", "husband_name", "booklet_number",
                    "cnic", "tracking_number",
                ],
            },
        },
        "required": ["mrz", "viz"],
    },
}


def extract(api_key: str, path: Path) -> dict:
    img_b64, media = encode_image(path)
    payload = {
        "model": MODEL,
        "max_tokens": 800,
        "temperature": 0,
        "tools": [EXTRACTION_TOOL],
        "tool_choice": {"type": "tool", "name": "record_passport_extraction"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media,
                            "data": img_b64,
                        },
                    },
                ],
            },
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    backoff = 5.0
    for attempt in range(6):
        with httpx.Client(timeout=120) as cli:
            r = cli.post(API_URL, headers=headers, json=payload)
        if r.status_code == 200:
            break
        if r.status_code in (429, 529, 500, 502, 503, 504):
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after else backoff
            time.sleep(min(wait, 60))
            backoff = min(backoff * 1.6, 60)
            continue
        return {"error": f"{r.status_code}: {r.text[:300]}"}
    if r.status_code != 200:
        return {"error": f"{r.status_code} after retries: {r.text[:300]}"}
    body = r.json()
    for block in body.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "record_passport_extraction":
            return block.get("input") or {}
    return {"error": "no tool_use block in response", "raw": json.dumps(body)[:500]}


def collect_images(src: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: list[Path] = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def file_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return f"{rel}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2

    images = collect_images(args.src)
    if not images:
        print(f"no images under {args.src}", file=sys.stderr)
        return 2
    print(f"found {len(images)} images", file=sys.stderr)

    random.seed(args.seed)
    if len(images) > args.sample:
        images = random.sample(images, args.sample)
    images.sort()
    print(f"sampling {len(images)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if args.out.exists():
        try:
            results = json.loads(args.out.read_text())
            print(f"resume: already have {len(results)}", file=sys.stderr)
        except Exception:
            results = {}

    # resume re-runs entries that previously errored out
    def is_done(fid: str) -> bool:
        v = results.get(fid)
        return bool(v) and "error" not in (v.get("fields") or {})

    todo = [p for p in images if not is_done(file_id(p, args.src))]
    print(f"to extract: {len(todo)}", file=sys.stderr)

    t0 = time.time()
    done = 0
    errs = 0
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futures = {ex.submit(extract, api_key, p): p for p in todo}
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            fid = file_id(p, args.src)
            try:
                fields = fut.result()
            except Exception as e:
                fields = {"error": f"exc: {e}"}
            results[fid] = {
                "filename": p.name,
                "agency": p.parent.name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                "fields": fields,
            }
            if "error" in fields:
                errs += 1
            done += 1
            if done % 8 == 0:
                args.out.write_text(json.dumps(results, indent=2))
                print(
                    f"  [{done}/{len(todo)}] {fid}  errs={errs}  "
                    f"elapsed={time.time()-t0:.0f}s",
                    file=sys.stderr,
                )

    args.out.write_text(json.dumps(results, indent=2))
    print(
        f"\ndone: wrote {len(results)} entries to {args.out} "
        f"({errs} errors, {time.time()-t0:.0f}s total)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
