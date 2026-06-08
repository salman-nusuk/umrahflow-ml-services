"""FastAPI passport OCR service.

POST /ocr  multipart with `file` field (image)            → JSON
POST /ocr  JSON {"url": "..."}                            → JSON
GET  /healthz                                             → {"ok": true}
"""
import asyncio, os, tempfile, time, urllib.request, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# load project-root .env (one level up from this service folder)
ENV_PATH = os.environ.get(
    "UMRAHFLOW_ENV_PATH",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env")),
)
if os.path.exists(ENV_PATH):
    for line in open(ENV_PATH):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

from run_pipeline_openai import process
from process_pdf import process_pdf

# Sentry — no-op when SENTRY_DSN_OCR (or fallback SENTRY_DSN) is unset.
_SENTRY_DSN = os.environ.get("SENTRY_DSN_OCR") or os.environ.get("SENTRY_DSN")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            traces_sample_rate=0.1,
            environment=os.environ.get("SENTRY_ENV", "production"),
        )
    except ImportError:
        pass

app = FastAPI(title="UmrahFlow OCR")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": os.environ.get("OPENAI_OCR_MODEL", "gpt-5.4-mini")}


class UrlRequest(BaseModel):
    url: str


def _run(path: str):
    t0 = time.time()
    r = process(path)
    r["t_total"] = round(time.time() - t0, 2)
    return r


@app.post("/ocr")
async def ocr_upload(file: UploadFile = File(None), payload: UrlRequest | None = None):
    if file is not None:
        suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            path = tmp.name
        try:
            # Off-load blocking vision+MRZ pipeline to a thread so the event
            # loop stays free for other concurrent /ocr requests. Without this,
            # 5 simultaneous extract_passport calls serialize at ~7s each.
            result = await asyncio.to_thread(_run, path)
            return JSONResponse(result)
        finally:
            os.unlink(path)
    raise HTTPException(400, "missing file")


@app.post("/ocr/url")
def ocr_url(req: UrlRequest):
    try:
        with urllib.request.urlopen(req.url, timeout=30) as r:
            data = r.read()
    except Exception as e:
        raise HTTPException(400, f"fetch failed: {e}")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return JSONResponse(_run(path))
    finally:
        os.unlink(path)


@app.post("/pdf")
async def pdf(file: UploadFile = File(...)):
    """Render every page of the uploaded PDF, classify each, and run the
    passport pipeline on every page that's classified as 'passport'.
    Voucher pages are flagged but not parsed here."""
    if not (file.content_type or "").lower().startswith("application/pdf") and \
       not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "expected a PDF file")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        # Same off-load reasoning as /ocr — process_pdf is sync + slow.
        result = await asyncio.to_thread(process_pdf, path)
        return JSONResponse(result)
    finally:
        os.unlink(path)
