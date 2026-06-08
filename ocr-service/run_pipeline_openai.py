"""Vision-primary passport OCR with MRZ verification.

Vision (GPT-5.4) extracts every canonical passport field. fastmrz runs in
parallel as an integrity check only — when MRZ succeeds and matches vision,
the passport is `verified`; otherwise it's saved but flagged for L1 review.
"""
import sys, os, json, base64, time, urllib.request, glob, re
from concurrent.futures import ThreadPoolExecutor
from fastmrz import FastMRZ

# load OPENAI_API_KEY from project .env
ENV_PATH = "/Users/afaqahmad/Documents/umrahflow-dashboard/umrahflow/.env"
if os.path.exists(ENV_PATH):
    for line in open(ENV_PATH):
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
            break

API_KEY = os.environ.get("OPENAI_API_KEY")
assert API_KEY, "OPENAI_API_KEY not set"

MODEL = os.environ.get("OPENAI_OCR_MODEL", "gpt-5.4-mini")

VIZ_PROMPT = """Extract every field from a Pakistani passport bio page. You are
the PRIMARY extractor — your output is what gets saved. Return ONLY the JSON
object below — no markdown, no commentary.

LAYOUT (Pakistani Machine Readable Passport, ICAO 9303):

  TOP-RIGHT corner: a serial like M9161105 (letter + 7 digits) — Booklet Number.
  Different from the passport number.

  BIO BLOCK (next to the photo): English-over-Urdu labels —
    Surname, Given Names, Nationality (PAKISTANI), Date of Birth, Sex (M/F)
    Place of Birth   -> CITY only (GUJRAT, LAHORE, etc.) — never PAKISTAN
    Place of Issue   -> usually the word PAKISTAN
    Date of Issue    -> DD MMM YYYY on the bio page
    Date of Expiry   -> DD MMM YYYY
    Passport Number  -> 8-9 chars, letter(s) + digits (e.g. BP0138231, FY1755541)

  BELOW the bio block (or right side):
    Father Name / Father's Name
    Husband Name / Spouse Name (married women)
    Identity Number / NIC / CNIC  -> 13 digits, format 12345-1234567-1
    Tracking No. / Citizenship Number  -> 11 digits (NOT the CNIC)

FIELD RULES:

surname / given_names
  EXACTLY as printed on the bio page. ALL CAPS. No leading/trailing spaces.
  given_names may be multi-word (e.g. "MUHAMMAD ALI HASSAN"). Do not invent.

passport_number
  Look UNDER the photo or to the right of "Passport Number". 8-9 characters,
  starts with 1-2 letters then digits (e.g. BP0138231, M9382289, TK5163901).
  Count carefully — small fonts on hotel-quality scans drop digits. Output
  exactly what's printed, no spaces.

date_of_birth / date_of_expiry / date_of_issue
  Output as YYYY-MM-DD. The bio page prints DD MMM YYYY (e.g. 14 AUG 1978).

gender
  Single character "M" or "F".

nationality
  "PAKISTANI" (or strictly what's printed). Country code: "PAK".

place_of_birth
  CITY ONLY: GUJRAT, LAHORE, KARACHI, FAISALABAD, etc. Strip ", PAK".

place_of_issue
  Exactly what's printed for "Place of Issue" / "Issuing Authority". Usually
  "PAKISTAN" — sometimes a city. Output verbatim.

father_name / husband_name
  Whichever label is printed. If "Husband Name", put it in husband_name (not
  father_name). If both visible, fill both. Output exactly what's printed.

booklet_number
  Top-right corner serial: letter + 7 digits (M9161105, A1234567). NOT the
  passport_number.

cnic
  13-digit Pakistani CNIC. Format strictly as 12345-1234567-1 (5-7-1 hyphens).
  If printed without hyphens, INSERT them. The MRZ optional-data field on
  Pakistani passports IS the CNIC — if you see 13 digits there and nowhere
  else on the bio page, use that (formatted with hyphens).

tracking_number
  11-digit number labeled "Tracking Number" or "Citizenship Number". NOT the
  CNIC (13 digits) and NOT the passport number. Plain 11 digits, no spaces.

If a field is genuinely not visible or unreadable, set it to null. Do not
guess. Do not output the country name in place_of_birth.

Output JSON shape (all fields required, use null when not present):
{
  "surname": null,
  "given_names": null,
  "passport_number": null,
  "date_of_birth": null,
  "date_of_expiry": null,
  "date_of_issue": null,
  "gender": null,
  "nationality": "PAKISTANI",
  "place_of_birth": null,
  "place_of_issue": null,
  "father_name": null,
  "husband_name": null,
  "booklet_number": null,
  "cnic": null,
  "tracking_number": null
}"""

mrz_engine = FastMRZ()


def run_mrz(path):
    t0 = time.time()
    try:
        return mrz_engine.get_details(path), time.time() - t0
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}, time.time() - t0


def run_viz(path):
    t0 = time.time()
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VIZ_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}},
            ],
        }],
        "response_format": {"type": "json_object"},
    }
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return {"data": json.loads(content), "usage": usage}, time.time() - t0
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}, time.time() - t0
    except Exception as e:
        return {"error": str(e)}, time.time() - t0


def _norm_str(s):
    if not s:
        return None
    return re.sub(r"\s+", " ", str(s).strip().upper())


def _norm_date(s):
    """Normalise 'YYYY-MM-DD' or 'YYMMDD' or 'DD MMM YYYY' to YYYY-MM-DD."""
    if not s:
        return None
    s = str(s).strip()
    # YYMMDD (MRZ format)
    if re.fullmatch(r"\d{6}", s):
        yy, mm, dd = s[:2], s[2:4], s[4:6]
        # MRZ year: assume 19YY if YY > current year + 5, else 20YY
        # crude but fine for passports
        yyyy = int(yy)
        cur = time.localtime().tm_year % 100
        century = 1900 if yyyy > cur + 5 else 2000
        return f"{century + yyyy:04d}-{mm}-{dd}"
    # already YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return s  # leave as-is; comparison may flag


def _verify(primary: dict, mrz: dict) -> dict:
    """Compare vision-primary fields with MRZ. Returns mismatches list and
    overall verified bool."""
    if mrz.get("status") != "SUCCESS":
        return {
            "status": mrz.get("status") or "FAIL",
            "parsed": None,
            "mismatches": [],
            "reason": "MRZ unreadable",
        }

    parsed = {
        "surname": _norm_str(mrz.get("surname")),
        "given_names": _norm_str(mrz.get("given_name")),
        "passport_number": _norm_str(mrz.get("document_number")),
        "date_of_birth": _norm_date(mrz.get("birth_date")),
        "date_of_expiry": _norm_date(mrz.get("expiry_date")),
        "gender": _norm_str(mrz.get("sex")),
    }

    checks = [
        ("passport_number", _norm_str(primary.get("passport_number")), parsed["passport_number"]),
        ("surname",          _norm_str(primary.get("surname")),         parsed["surname"]),
        ("date_of_birth",    _norm_date(primary.get("date_of_birth")),  parsed["date_of_birth"]),
        ("date_of_expiry",   _norm_date(primary.get("date_of_expiry")), parsed["date_of_expiry"]),
        ("gender",           _norm_str(primary.get("gender")),          parsed["gender"]),
    ]
    mismatches = []
    for field, viz_val, mrz_val in checks:
        if viz_val and mrz_val and viz_val != mrz_val:
            mismatches.append(f"{field}: VIZ={viz_val} vs MRZ={mrz_val}")

    return {
        "status": "SUCCESS",
        "parsed": parsed,
        "mismatches": mismatches,
        "reason": None if not mismatches else "MRZ disagrees with vision on " + ", ".join(m.split(":")[0] for m in mismatches),
    }


def process(path):
    """Vision-primary OCR with MRZ verification.

    Output shape:
      {
        primary: { full passport extracted by vision },
        mrz_check: { status, parsed, mismatches, reason },
        verified: bool,  # primary saved with confidence when true
        viz_error: str | None,
        t_viz, t_mrz, t_wall
      }
    """
    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_mrz = ex.submit(run_mrz, path)
        f_viz = ex.submit(run_viz, path)
        mrz, t_mrz = f_mrz.result()
        viz, t_viz = f_viz.result()

    primary = viz.get("data", {}) if "data" in viz else {}
    # canonicalise the surface fields so downstream sees consistent casing
    if primary.get("surname"):
        primary["surname"] = _norm_str(primary["surname"])
    if primary.get("given_names"):
        primary["given_names"] = _norm_str(primary["given_names"])
    if primary.get("passport_number"):
        primary["passport_number"] = _norm_str(primary["passport_number"])
    if primary.get("gender"):
        primary["gender"] = _norm_str(primary["gender"])
    for d in ("date_of_birth", "date_of_expiry", "date_of_issue"):
        if primary.get(d):
            primary[d] = _norm_date(primary[d])

    mrz_check = _verify(primary, mrz)
    verified = mrz_check["status"] == "SUCCESS" and not mrz_check["mismatches"]

    return {
        "file": os.path.basename(path),
        "primary": primary,
        "mrz_check": mrz_check,
        "verified": verified,
        "viz_error": viz.get("error"),
        "viz_usage": viz.get("usage"),
        "t_mrz": round(t_mrz, 2),
        "t_viz": round(t_viz, 2),
        "t_wall": round(time.time() - t, 2),
    }


def main():
    samples_glob = sys.argv[1] if len(sys.argv) > 1 else "/Users/afaqahmad/Downloads/ses 11111/**/*.jpeg"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paths = sorted(glob.glob(samples_glob, recursive=True))[:n]
    print(f"# model={MODEL}  benchmarking {len(paths)} samples\n", flush=True)

    results = []
    total = time.time()
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
        r = process(p)
        results.append(r)
        usage = r.get("viz_usage") or {}
        v = "✓" if r["verified"] else "?"
        print(f"   {v} mrz={r['mrz_check']['status']} mism={len(r['mrz_check']['mismatches'])} t_viz={r['t_viz']}s t_wall={r['t_wall']}s", flush=True)

    verified = sum(1 for r in results if r["verified"])
    avg_wall = sum(r["t_wall"] for r in results) / len(results)
    print("\n# summary")
    print(f"verified: {verified}/{len(results)}")
    print(f"avg wall: {avg_wall:.2f}s")
    print(f"total:    {time.time()-total:.1f}s")

    with open("benchmark_openai.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nfull results -> benchmark_openai.json")


if __name__ == "__main__":
    main()
