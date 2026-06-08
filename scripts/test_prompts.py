"""A/B test for the VIZ prompt. Runs N images through the production OpenAI
model with two different prompts and reports per-field recall vs the Sonnet
4.6 ground truth.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import httpx

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

OLD_PROMPT = """You are extracting structured fields from a Pakistani passport bio page (machine-readable
travel document, ICAO 9303). Read carefully, then return ONLY a single JSON object.

Field-by-field instructions:

- place_of_birth: The CITY printed next to "Place of Birth". Must be a Pakistani city
  (e.g. LAHORE, KARACHI, ISLAMABAD, FAISALABAD, GUJRAT, BAHAWALPUR, PESHAWAR, MULTAN,
  RAWALPINDI). NEVER output "PAK", "PAKISTAN", or a country name. Preserve original casing.

- place_of_issue: Value next to "Place of Issue" or "Issuing Authority".
  On Pakistani passports this is typically the literal word "PAKISTAN".
  Output exactly what is printed (PAKISTAN, or a city like ISLAMABAD).

- date_of_issue: Date next to "Date of Issue", in YYYY-MM-DD format.

- father_name: Value next to the label "Father Name" (sometimes "Father's Name" or
  in Urdu followed by an English line). Do NOT copy the holder's own surname or
  given name. If the label is not visible, return null.

- husband_name: Value next to "Husband Name" / "Husband Names" / "Spouse Name".
  Pakistani passports for married women typically show this in place of (or
  alongside) Father Name. Extract whatever is printed; null if not visible.

- tracking_number: 11-digit number next to "Tracking Number" or "Citizenship
  Number". Distinct from CNIC. Null if not visible.

- booklet_number: Serial printed at the top-right corner of the bio page,
  starting with a letter (e.g. M9161105, A1234567). NOT the same as the
  passport number in the MRZ.

- cnic: Pakistani national ID number. Exactly 13 digits, formatted
  XXXXX-XXXXXXX-X (5-7-1). Reject anything that is not exactly 13 digits.
  May appear as "Tracking Number" or unlabelled. If only the MRZ optional-data
  number is visible, leave null (we'll cross-reference separately).

If a field is not clearly readable, output null. DO NOT guess. Better to return
null than wrong data — this feeds a visa application system.

Return only this JSON shape (no commentary, no markdown):
{
  "place_of_birth": null,
  "place_of_issue": null,
  "date_of_issue": null,
  "father_name": null,
  "husband_name": null,
  "booklet_number": null,
  "cnic": null,
  "tracking_number": null
}"""

NEW_PROMPT = """Extract these fields from a Pakistani passport bio page. Return ONLY the JSON
object below — no markdown, no commentary.

LAYOUT REFERENCE (Pakistani Machine Readable Passport — ICAO 9303 compliant):

  TOP-RIGHT corner: a serial like M9161105 (letter + 7 digits) — this is
  Booklet Number. Different from the passport number printed under the photo.

  BIO BLOCK (next to the photo): labels in English over Urdu —
    Surname, Given Names, Nationality, Date of Birth, Sex
    Place of Birth          -> CITY only (GUJRAT, LAHORE, etc.) never PAKISTAN
    Place of Issue          -> usually the word PAKISTAN, sometimes a city
    Date of Issue           -> on the bio page in DD MMM YYYY format
    Date of Expiry
    Passport Number         -> matches the MRZ document number

  BELOW the bio block, OR on the right side, OR labeled in Urdu/English:
    Father Name / Father's Name / Husband Name / Spouse Name
    Identity Number / NIC / CNIC          <- 13 digits, format 12345-1234567-1
    Tracking No. / Tracking Number / Citizenship Number  <- 11 digits

FIELD RULES:

place_of_birth
  The CITY printed for Place of Birth. e.g. "GUJRAT", "LAHORE", "KARACHI".
  Never the country. Strip trailing ", PAK" if present.

place_of_issue
  Output what's literally printed for Place of Issue / Issuing Authority.
  Usually "PAKISTAN". Sometimes a city. Output exactly what you see.

date_of_issue
  The "Date of Issue" printed on the bio page (NOT the date_of_birth and NOT
  the date_of_expiry). On Pakistani passports it's printed in DD MMM YYYY
  form (e.g. 16 OCT 2024 — three-letter month). Output as YYYY-MM-DD.

father_name
  Value next to "Father Name" / "Father's Name". For married women this slot
  may instead be used for the husband's name — if the label is "Husband Name"
  put it in husband_name not father_name. Output exactly what's printed.

husband_name
  Value next to "Husband Name" / "Husband's Name" / "Spouse Name". Same rules.

booklet_number
  The serial printed in the TOP-RIGHT corner — starts with a letter, then 7
  digits (e.g. M9161105, A1234567, R9272820). This is NOT the passport number.

cnic
  The 13-digit Pakistani National Identity Card number. Look for it labeled
  "Identity Number", "NIC", "CNIC", "ID Card", or printed below the bio
  block. Format strictly as 12345-1234567-1 (5-7-1 with hyphens). If the
  number is printed without hyphens, INSERT them — output 12345-1234567-1.
  Only return null if you cannot find any 13-digit number associated with
  the passport holder. The MRZ optional-data field on Pakistani passports
  IS the CNIC — if you can read 13 digits there and nowhere else, use that
  (formatted with hyphens).

tracking_number
  An 11-digit number labeled "Tracking Number", "Tracking No.", or
  "Citizenship Number". Do NOT confuse with CNIC (which is 13 digits) or
  the passport number. Return as 11 digits, no spaces.

Output JSON shape (all fields required, use null when not present):
{
  "place_of_birth": null,
  "place_of_issue": null,
  "date_of_issue": null,
  "father_name": null,
  "husband_name": null,
  "booklet_number": null,
  "cnic": null,
  "tracking_number": null
}"""


def call_openai(api_key: str, model: str, prompt: str, path: Path) -> dict:
    img = base64.b64encode(path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            ],
        }],
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=120) as cli:
        r = cli.post(OPENAI_URL, headers=headers, json=payload)
    if r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:200]}"}
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return {"error": f"parse: {e}"}


def norm_cnic(v):
    if not v:
        return None
    s = "".join(c for c in str(v) if c.isdigit())
    return f"{s[:5]}-{s[5:12]}-{s[12:13]}" if len(s) == 13 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing", file=sys.stderr)
        return 2

    ref = json.loads(args.reference.read_text())
    # Pick samples where Sonnet read both cnic and tracking_number — that's
    # where we have the strongest signal for whether the prompt is the limiter.
    candidates = []
    for k, v in ref.items():
        viz = (v.get("fields", {}) or {}).get("viz", {}) or {}
        if viz.get("cnic") or viz.get("tracking_number"):
            candidates.append(k)
    import random
    random.seed(args.seed)
    samples = random.sample(candidates, min(args.n, len(candidates)))
    print(f"testing {len(samples)} images on model={args.model}", file=sys.stderr)

    results: dict[str, dict] = {}
    def work(fid: str, prompt_name: str, prompt: str):
        return fid, prompt_name, call_openai(api_key, args.model, prompt, args.src / fid)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futs = []
        for fid in samples:
            futs.append(ex.submit(work, fid, "old", OLD_PROMPT))
            futs.append(ex.submit(work, fid, "new", NEW_PROMPT))
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            fid, name, out = fut.result()
            results.setdefault(fid, {})[name] = out
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(futs)} elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    # Compare
    fields = ["place_of_birth", "place_of_issue", "date_of_issue", "father_name",
              "husband_name", "booklet_number", "cnic", "tracking_number"]
    stats = {p: {f: {"refnonnull": 0, "match": 0, "prodnull": 0} for f in fields}
             for p in ("old", "new")}
    for fid in samples:
        rviz = (ref[fid].get("fields", {}) or {}).get("viz", {}) or {}
        for prompt_name in ("old", "new"):
            out = results[fid].get(prompt_name) or {}
            if "error" in out:
                continue
            for f in fields:
                ref_v = rviz.get(f)
                if not ref_v:
                    continue
                stats[prompt_name][f]["refnonnull"] += 1
                got = out.get(f)
                # cnic: compare canonical form
                if f == "cnic":
                    a, b = norm_cnic(ref_v), norm_cnic(got)
                else:
                    a = (str(ref_v).upper().strip() if ref_v else None)
                    b = (str(got).upper().strip() if got else None)
                if not got:
                    stats[prompt_name][f]["prodnull"] += 1
                elif a == b:
                    stats[prompt_name][f]["match"] += 1

    print(f"\n# A/B prompt test on {len(samples)} images, model={args.model}\n")
    print(f"{'field':22s} {'OLD match/ref':>16s}  {'NEW match/ref':>16s}  {'OLD null':>9s}  {'NEW null':>9s}")
    for f in fields:
        o = stats["old"][f]
        n = stats["new"][f]
        if o["refnonnull"] == 0:
            continue
        o_pct = o["match"] / o["refnonnull"] * 100
        n_pct = n["match"] / n["refnonnull"] * 100
        delta = n_pct - o_pct
        marker = "↑" if delta > 5 else "↓" if delta < -5 else " "
        print(f"{f:22s} "
              f"{o['match']:3d}/{o['refnonnull']:<3d} ({o_pct:5.1f}%)  "
              f"{n['match']:3d}/{n['refnonnull']:<3d} ({n_pct:5.1f}%) {marker}  "
              f"{o['prodnull']:9d}  {n['prodnull']:9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
