"""Generate synthetic test fixtures for scenarios.yml that need crafted inputs.
Run from this directory: `python3 _build_synth.py`. Outputs are ignored by
manifest.yml — these are just blobs that match the paths scenarios reference.

What we generate (intent, not perfection):
  corrupt/blank.jpg                       — solid-white image (OCR returns empty)
  AGOG/multi-passports.pdf                — agent_A's 5 passport JPGs in one PDF
  AGOG/combined-voucher-and-passports.pdf — agent_A voucher.pdf followed by 5 jpgs
  edge/passport-empty-pno.jpg             — agent_A/p1 with passport-number area whited out
  stray/unknown_passport.jpg              — copy of agent_B/p5 (not in agent_A's mutamers)
  shared/shared_passport.jpg              — copy of agent_A/p1 (used as "duplicate on both")
  edge/voucher-no-mutamers.pdf            — text-rendered voucher header, no mutamer rows
  cross-agent/AB1234567.jpg               — text-rendered passport-like card with PNO=AB1234567
  cross-agent/UB-T999002-voucher.pdf      — text-rendered voucher with UB-T999002
  blacklist/AB1234567.jpg                 — same synthetic passport, different folder

The cross-agent/blacklist/voucher-no-mutamers fixtures are TEXT-rendered (not real
photos). The vision model in extract_passport / extract_voucher reads text just fine —
they're realistic enough for path-coverage tests, not OCR-quality benchmarks.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
A = HERE / "agent_A"
B = HERE / "agent_B"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def write_blank() -> None:
    HERE.joinpath("corrupt").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1024, 768), "white").save(HERE / "corrupt" / "blank.jpg", "JPEG", quality=85)
    # Some scenarios reference blank2.jpg to send two corrupt frames in one burst.
    Image.new("RGB", (800, 1200), (245, 245, 245)).save(HERE / "corrupt" / "blank2.jpg", "JPEG", quality=85)


def write_multi_passports_pdf() -> None:
    out = HERE / "AGOG" / "multi-passports.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [Image.open(A / f).convert("RGB") for f in sorted(A.glob("p*.jpg"))]
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:])


def write_combined_pdf() -> None:
    """Concatenate agent_A/voucher.pdf with 5 passport JPGs into one PDF."""
    out = HERE / "AGOG" / "combined-voucher-and-passports.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    # PIL can't read existing PDFs. Render the voucher PDF's "vibe" as a synthetic
    # text page, then append the real passport jpgs. Good enough for the test —
    # the agent should classify page 0 as voucher and pages 1..5 as passports.
    voucher_page = _render_voucher_page("UB-181535", "Mahboob Alam Muhammad Yaqoob",
                                        ["FY1755541", "JT3963061", "BP0138231",
                                         "QQ4912891", "TK5163901"])
    pages = [voucher_page] + [Image.open(A / f).convert("RGB") for f in sorted(A.glob("p*.jpg"))]
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:])


def write_passport_empty_pno() -> None:
    """Take p1, white-out the rough passport-number region. The agent should
    extract the rest of the fields but pno stays empty after .strip()."""
    out = HERE / "edge" / "passport-empty-pno.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(A / "p1_FY1755541_mahboob.jpg").convert("RGB")
    w, h = img.size
    # Cover the full top-right quadrant + the MRZ band at the bottom — the
    # passport-number lives in both. (Approximate rectangles for a typical
    # Pakistani booklet scan; precision doesn't matter — we just want the OCR
    # path to fail to recover a passport_number.)
    draw = ImageDraw.Draw(img)
    draw.rectangle([w * 0.55, h * 0.05, w * 0.99, h * 0.30], fill="white")
    draw.rectangle([0, h * 0.85, w, h], fill="white")
    img.save(out, "JPEG", quality=85)


def write_unknown_passport() -> None:
    out = HERE / "stray" / "unknown_passport.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    # A passport not on agent_A's voucher (we'll send agent_A's voucher in the
    # same scenario, so any agent_B passport qualifies as "stray").
    shutil.copy2(B / "p5_QW3959892_naseem.jpg", out)


def write_shared_passport() -> None:
    out = HERE / "shared" / "shared_passport.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(A / "p1_FY1755541_mahboob.jpg", out)


def _render_voucher_page(ub: str, family_head: str, mutamers: list[str]) -> Image.Image:
    """Draw a plain text 'voucher' card. Vision model reads this fine."""
    img = Image.new("RGB", (1240, 1754), "white")  # ~A4 @ 150dpi
    d = ImageDraw.Draw(img)
    title = _font(44)
    body = _font(28)
    d.text((80, 80), "UMRAH VOUCHER", fill="black", font=title)
    d.text((80, 180), f"UB Number: {ub}", fill="black", font=body)
    d.text((80, 230), f"Family Head: {family_head}", fill="black", font=body)
    d.text((80, 280), f"Agency: Test Agency", fill="black", font=body)
    d.text((80, 360), "Expected Mutamers:", fill="black", font=body)
    for i, m in enumerate(mutamers):
        d.text((100, 420 + i * 40), f"{i + 1}. Passport {m}", fill="black", font=body)
    return img


def write_voucher_no_mutamers() -> None:
    out = HERE / "edge" / "voucher-no-mutamers.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    page = _render_voucher_page("UB-T700001", "No Mutamer Test", [])
    page.save(out, "PDF")


def write_synthetic_passport(pno: str, target: Path) -> None:
    """A passport-shaped card with the given passport number prominent. Won't
    pass MRZ verification — extract_passport only tags it `verified=False`,
    which is fine for the cross-agent / blacklist scenarios."""
    target.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 700), "white")
    d = ImageDraw.Draw(img)
    title = _font(48)
    body = _font(32)
    mono = _font(26)
    d.rectangle([20, 20, 1004, 680], outline="black", width=4)
    d.text((60, 50), "ISLAMIC REPUBLIC OF PAKISTAN", fill="black", font=title)
    d.text((60, 130), "PASSPORT", fill="black", font=title)
    d.text((60, 230), f"Passport No: {pno}", fill="black", font=body)
    d.text((60, 290), "Surname: TEST", fill="black", font=body)
    d.text((60, 350), "Given Names: SYNTHETIC", fill="black", font=body)
    d.text((60, 410), "Date of Birth: 01 JAN 1990", fill="black", font=body)
    d.text((60, 470), "Sex: M", fill="black", font=body)
    d.text((60, 530), "Date of Expiry: 01 JAN 2030", fill="black", font=body)
    # Synthetic MRZ (won't pass checksum — that's intentional)
    mrz1 = f"P<PAK{('TEST<<SYNTHETIC' + '<' * 30)[:39]}"
    mrz2 = f"{pno}<7PAK9001017M3001017<<<<<<<<<<<<<<00"
    d.text((60, 600), mrz1[:44], fill="black", font=mono)
    d.text((60, 632), mrz2[:44], fill="black", font=mono)
    img.save(target, "JPEG", quality=85)


def write_synthetic_voucher_pdf(ub: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    page = _render_voucher_page(ub, "Test Family Head",
                                ["AB1234567", "CD7654321", "EF1122334"])
    page.save(target, "PDF")


def main() -> None:
    write_blank()
    write_multi_passports_pdf()
    write_combined_pdf()
    write_passport_empty_pno()
    write_unknown_passport()
    write_shared_passport()
    write_voucher_no_mutamers()

    write_synthetic_passport("AB1234567", HERE / "cross-agent" / "AB1234567.jpg")
    write_synthetic_passport("AB1234567", HERE / "blacklist" / "AB1234567.jpg")
    write_synthetic_voucher_pdf("UB-T999002", HERE / "cross-agent" / "UB-T999002-voucher.pdf")

    print("ok")


if __name__ == "__main__":
    main()
