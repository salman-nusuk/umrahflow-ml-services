"""Multi-page PDF passport extraction.

Renders each PDF page to an image, vision-classifies each page (passport /
voucher / other), and runs the full passport pipeline on every passport page.
Voucher pages are flagged but not parsed here — the agent's existing
extract_voucher tool handles voucher PDFs.
"""
import os
import time
import json
import base64
import tempfile
import urllib.request

import pypdfium2 as pdfium

from run_pipeline_openai import process as process_passport_image

API_KEY = os.environ.get("OPENAI_API_KEY")
CLASSIFY_MODEL = os.environ.get("OPENAI_CLASSIFY_MODEL", "gpt-5.4-nano")

# Render PDF pages at 200 DPI — enough for MRZ readability.
PDF_DPI = int(os.environ.get("PDF_RENDER_DPI", "200"))
# Cap pages to avoid runaway processing on a 200-page accidental upload.
MAX_PAGES = int(os.environ.get("PDF_MAX_PAGES", "30"))


def _render_pages(pdf_path: str) -> list[str]:
    """Render every page of `pdf_path` to a JPEG and return the temp paths."""
    paths: list[str] = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        scale = PDF_DPI / 72.0
        for i, page in enumerate(pdf):
            if i >= MAX_PAGES:
                break
            pil_image = page.render(scale=scale).to_pil()
            tmp = tempfile.NamedTemporaryFile(suffix=f"_p{i+1}.jpg", delete=False)
            pil_image.convert("RGB").save(tmp.name, "JPEG", quality=92)
            paths.append(tmp.name)
            page.close()
    finally:
        pdf.close()
    return paths


def _classify_page(image_path: str) -> str:
    """Single-page classifier — same prompt as classify_media."""
    if not API_KEY:
        return "other"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": CLASSIFY_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Classify this image. Reply with one word, lowercase: "
                    "passport (Pakistani passport bio page), "
                    "voucher (visa/hotel/booking voucher), "
                    "id_card (CNIC or other ID), "
                    "payment_proof, other, or unreadable."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_completion_tokens": 8,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        return resp["choices"][0]["message"]["content"].strip().lower().strip(".")
    except Exception:
        return "other"


def _extract_passport_page(image_path: str) -> dict:
    """Vision-primary extraction with MRZ verification — same shape as
    run_pipeline_openai.process()."""
    return process_passport_image(image_path)


def process_pdf(pdf_path: str) -> dict:
    """Render the PDF, classify each page, run passport OCR on passport pages."""
    t0 = time.time()
    page_paths = _render_pages(pdf_path)
    pages: list[dict] = []
    try:
        for idx, ppath in enumerate(page_paths, start=1):
            kind = _classify_page(ppath)
            entry: dict = {"page": idx, "kind": kind}
            if kind == "passport":
                entry.update(_extract_passport_page(ppath))
            pages.append(entry)
    finally:
        for p in page_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return {
        "page_count": len(pages),
        "pages": pages,
        "t_total": round(time.time() - t0, 2),
    }
