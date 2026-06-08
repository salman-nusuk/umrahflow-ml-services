"""Compare production OCR results against the Sonnet 4.6 ground-truth reference
and print a per-field accuracy report.

Usage:
    OPENAI_API_KEY=sk-... \
    python scripts/compare_ocr.py \
        --src '/Users/afaqahmad/Downloads/ses 11111' \
        --reference tests/fixtures/ground_truth.json \
        --production tests/fixtures/production.json \
        --report tests/fixtures/report.md \
        --concurrency 8

The script:
1. Loads the reference (ground-truth) JSON.
2. Runs the production OCR pipeline on the same files (writes production.json).
3. Computes per-field exact-match and fuzzy-match accuracy.
4. Writes a markdown report and exits non-zero if any field falls below the
   threshold defined in PASS_THRESHOLDS.
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# Same prompt used by the production OCR pipeline (mirrors run_pipeline_openai.py
# at the time of reference build). We only need it for the production run.
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.4-mini"

VIZ_PROMPT = """Extract these visual-zone fields from the Pakistani passport
image. Return only this JSON, no commentary:
{
  "place_of_birth": null,
  "place_of_issue": null,
  "date_of_issue": null,
  "father_name": null,
  "husband_name": null,
  "booklet_number": null,
  "cnic": null,
  "tracking_number": null
}

Rules:
- place_of_birth: city only (e.g. "GUJRAT", not "GUJRAT, PAK").
- place_of_issue: literal text (may be "PAKISTAN" or a city). Never output a country code.
- date_of_issue: YYYY-MM-DD.
- father_name / husband_name: only if printed.
- booklet_number: serial top-right starting with a letter.
- cnic: 13-digit Pakistani ID, formatted XXXXX-XXXXXXX-X.
- tracking_number: 11-digit "Tracking Number" or "Citizenship Number".

Null is better than wrong."""

PASS_THRESHOLDS = {
    "mrz.passport": 0.95,
    "mrz.dob": 0.95,
    "mrz.expiry": 0.95,
    "mrz.sex": 0.95,
    "mrz.name": 0.85,
    "viz.place_of_issue": 0.80,
    "viz.date_of_issue": 0.80,
    "viz.cnic": 0.80,
    "viz.father_name": 0.70,
    "viz.husband_name": 0.70,
    "viz.place_of_birth": 0.70,
    "viz.booklet_number": 0.70,
    "viz.tracking_number": 0.70,
}


def norm(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    s = re.sub(r"[\s,]+$", "", s)
    s = re.sub(r",\s*PAK(ISTAN)?$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s or None


def name_similarity(a: str, b: str) -> float:
    """Simple token-set similarity for names (handles spelling/order variation)."""
    if not a or not b:
        return 1.0 if a == b else 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


FIELD_FAMILIES = {
    "name": "fuzzy",
    "father_name": "fuzzy",
    "husband_name": "fuzzy",
    "place_of_birth": "fuzzy",
    "place_of_issue": "fuzzy",
}


def field_match(field: str, ref, prod) -> str:
    """Return one of: 'match', 'mismatch', 'ref_null', 'prod_null', 'both_null'."""
    ref_n = norm(ref)
    prod_n = norm(prod)
    if ref_n is None and prod_n is None:
        return "both_null"
    if ref_n is None:
        return "ref_null"
    if prod_n is None:
        return "prod_null"
    if FIELD_FAMILIES.get(field) == "fuzzy":
        return "match" if name_similarity(ref_n, prod_n) >= 0.6 else "mismatch"
    return "match" if ref_n == prod_n else "mismatch"


def run_production_viz(api_key: str, path: Path) -> dict:
    import base64
    img_b64 = base64.b64encode(path.read_bytes()).decode()
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VIZ_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=120) as cli:
        r = cli.post(OPENAI_URL, headers=headers, json=payload)
    if r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:300]}"}
    try:
        text = r.json()["choices"][0]["message"]["content"]
        return json.loads(text)
    except Exception as e:
        return {"error": f"parse: {e}"}


def run_production_ocr(endpoint: str, path: Path) -> dict:
    """POST the image to the deployed OCR endpoint (which has rotation fallback,
    fastmrz, and GPT-5.4 Mini wired together). Returns {mrz, viz} or {error}.
    """
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        with httpx.Client(timeout=120) as cli:
            r = cli.post(endpoint, files=files)
    if r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:300]}"}
    body = r.json()
    return {
        "mrz": body.get("mrz") or {},
        "viz": body.get("viz") or {},
        "fallback_used": body.get("fallback_used", False),
    }


def file_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--production", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-production", action="store_true",
                    help="Reuse the existing production.json instead of regenerating.")
    ap.add_argument("--endpoint", default="https://passport.delveon.com/api/ocr",
                    help="Deployed OCR endpoint that has rotation fallback wired up.")
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"reference not found: {args.reference}", file=sys.stderr)
        return 2
    reference: dict = json.loads(args.reference.read_text())

    # Run production OCR for any file present in the reference.
    if not args.skip_production:
        production: dict = {}
        if args.production.exists():
            try:
                production = json.loads(args.production.read_text())
            except Exception:
                production = {}

        # Re-run any entries whose existing data is malformed (no real MRZ).
        def is_complete(entry: dict) -> bool:
            f = entry.get("fields", {})
            if "error" in f:
                return False
            mrz = f.get("mrz") or {}
            return any(mrz.get(k) for k in ("passport", "name", "dob"))

        todo = [
            k for k in reference.keys()
            if k not in production or not is_complete(production[k])
        ]
        print(f"reference={len(reference)} production_existing={len(production)} todo={len(todo)} endpoint={args.endpoint}", file=sys.stderr)

        def work(fid: str):
            path = args.src / fid
            return fid, run_production_ocr(args.endpoint, path)

        t0 = time.time()
        done = 0
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
            futures = {ex.submit(work, fid): fid for fid in todo}
            for fut in concurrent.futures.as_completed(futures):
                fid, fields = fut.result()
                production[fid] = {"fields": fields}
                done += 1
                if done % 8 == 0:
                    args.production.parent.mkdir(parents=True, exist_ok=True)
                    args.production.write_text(json.dumps(production, indent=2))
                    print(f"  [{done}/{len(todo)}] {fid}  elapsed={time.time()-t0:.0f}s",
                          file=sys.stderr)
        args.production.parent.mkdir(parents=True, exist_ok=True)
        args.production.write_text(json.dumps(production, indent=2))
        print(f"production OCR done in {time.time()-t0:.0f}s", file=sys.stderr)
    else:
        production = json.loads(args.production.read_text())

    # Compare. We measure accuracy ONLY over files where the reference has a
    # non-null value (otherwise the field tells us nothing about regression).
    field_paths = [
        ("mrz", "name"),
        ("mrz", "passport"),
        ("mrz", "dob"),
        ("mrz", "sex"),
        ("mrz", "expiry"),
        ("viz", "place_of_birth"),
        ("viz", "place_of_issue"),
        ("viz", "date_of_issue"),
        ("viz", "father_name"),
        ("viz", "husband_name"),
        ("viz", "booklet_number"),
        ("viz", "cnic"),
        ("viz", "tracking_number"),
    ]

    summary: dict = {}
    mismatches: list = []
    for ns, field in field_paths:
        key = f"{ns}.{field}"
        ref_present = match = mismatch = prod_null = 0
        for fid, ref_entry in reference.items():
            ref_fields = ref_entry.get("fields", {})
            if "error" in ref_fields:
                continue
            ref_val = (ref_fields.get(ns) or {}).get(field)
            prod_entry = production.get(fid, {}).get("fields", {})
            prod_val = (prod_entry.get(ns) or {}).get(field) if isinstance(prod_entry, dict) else None
            verdict = field_match(field, ref_val, prod_val)
            if verdict in ("both_null", "ref_null"):
                continue
            ref_present += 1
            if verdict == "match":
                match += 1
            elif verdict == "prod_null":
                prod_null += 1
            else:
                mismatch += 1
                if len(mismatches) < 50:
                    mismatches.append({
                        "file": fid,
                        "field": key,
                        "expected": ref_val,
                        "got": prod_val,
                    })
        accuracy = match / ref_present if ref_present > 0 else 1.0
        summary[key] = {
            "ref_present": ref_present,
            "match": match,
            "prod_null": prod_null,
            "mismatch": mismatch,
            "accuracy": accuracy,
            "threshold": PASS_THRESHOLDS.get(key),
        }

    # Render report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# OCR Comparison Report\n")
    lines.append(f"- Reference: `{args.reference}` ({len(reference)} entries)")
    lines.append(f"- Production: `{args.production}` ({len(production)} entries)")
    lines.append("")
    lines.append("## Per-field accuracy (ref-non-null)\n")
    lines.append("| Field | Refs | Match | Prod null | Mismatch | Accuracy | Threshold | Pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    failed = []
    for key, s in summary.items():
        threshold = s["threshold"]
        passes = threshold is None or s["accuracy"] >= threshold
        if not passes:
            failed.append(key)
        lines.append(
            f"| `{key}` | {s['ref_present']} | {s['match']} | {s['prod_null']} | "
            f"{s['mismatch']} | {s['accuracy']:.1%} | "
            f"{threshold if threshold is not None else '—'} | "
            f"{'✓' if passes else '✗'} |"
        )
    lines.append("")
    if mismatches:
        lines.append("## Sample mismatches\n")
        lines.append("| File | Field | Expected | Got |")
        lines.append("|---|---|---|---|")
        for m in mismatches[:30]:
            exp = (str(m["expected"]) or "").replace("|", "\\|")[:40]
            got = (str(m["got"]) or "").replace("|", "\\|")[:40]
            lines.append(f"| `{m['file']}` | `{m['field']}` | {exp} | {got} |")
        lines.append("")
    if failed:
        lines.append(f"\n**FAILED:** {len(failed)} field(s) below threshold: {', '.join(failed)}")
    else:
        lines.append("\n**ALL PASS**")
    args.report.write_text("\n".join(lines))
    print(args.report.read_text())

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
