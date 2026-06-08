"""Image preprocessing for Pakistani passport OCR.

Goals: rescue scans that fastmrz/VLM fail on due to skew, rotation, low contrast,
or oversize. CPU-only OpenCV; ~50ms per image.
"""
import cv2
import numpy as np
from PIL import Image
import io

MAX_LONG_EDGE = 1600


def _to_cv(img):
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # already numpy/cv


def _from_cv(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def rotate(cv_img, deg):
    if deg == 0:
        return cv_img
    if deg == 90:
        return cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(cv_img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(deg)


def deskew(cv_img, max_angle=15.0):
    """Estimate page tilt from text lines and counter-rotate. Skips if estimated
    angle is tiny or out of plausible range."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return cv_img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return cv_img
    h, w = cv_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def boost_contrast(cv_img):
    """CLAHE on the L channel — recovers faded MRZ bands without overcooking
    the photo region."""
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def resize_cap(cv_img, max_edge=MAX_LONG_EDGE):
    h, w = cv_img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return cv_img
    scale = max_edge / long_edge
    return cv2.resize(cv_img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def preprocess(image_path: str, out_path: str | None = None) -> str:
    """Read image_path, apply pipeline, save to out_path (or /tmp/<name>_pp.jpg).
    Returns the saved path."""
    import os, tempfile
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise ValueError(f"could not read {image_path}")
    cv_img = deskew(cv_img)
    cv_img = boost_contrast(cv_img)
    cv_img = resize_cap(cv_img)
    if out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(tempfile.gettempdir(), f"{base}_pp.jpg")
    cv2.imwrite(out_path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def make_rotation_variants(image_path: str) -> list[str]:
    """Save 0°/90°/180°/270° rotations as separate files. Returns paths."""
    import os, tempfile
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise ValueError(f"could not read {image_path}")
    cv_img = boost_contrast(resize_cap(cv_img))
    base = os.path.splitext(os.path.basename(image_path))[0]
    paths = []
    for deg in (0, 90, 180, 270):
        rotated = rotate(cv_img, deg)
        p = os.path.join(tempfile.gettempdir(), f"{base}_r{deg}.jpg")
        cv2.imwrite(p, rotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    out = preprocess(sys.argv[1])
    print(out)
