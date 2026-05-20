"""
Lambda: edit
Core image processing worker.
Reads RAW/JPEG from S3, applies edits via rawpy + numpy, writes JPEG to output bucket.

Supported params (all floats, sent from the browser):
  exposure     : EV stops  (-3 … +3)
  highlights   : (-100 … 100)
  shadows      : (-100 … 100)
  blacks        : (-100 … 100)
  whites        : (-100 … 100)
  temp          : colour temperature shift (-100 … 100)
  tint          : green–magenta tint shift (-100 … 100)
  vibrance      : (-100 … 100)
  saturation    : (-100 … 100)
  sharpness     : (0 … 150)
  nr            : noise reduction (0 … 100)
  quality       : JPEG output quality (60 … 100)
"""
import io
import json
import os
import time
import tempfile

import boto3
import numpy as np
from PIL import Image
from botocore.exceptions import ClientError

# rawpy is installed via Lambda Layer (Docker-packaged with libraw)
try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

s3  = boto3.client("s3")
ddb = boto3.resource("dynamodb")

UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
TABLE_NAME    = os.environ["JOBS_TABLE"]

RAW_EXTENSIONS = {"raw", "nef", "cr2", "cr3", "arw", "dng", "orf", "rw2"}


# ─────────────────────────── helpers ────────────────────────────────────────

def _cors(body, status=200):
    return {
        "statusCode": status,
        "headers": {
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }


def _update_job(job_id: str, **kwargs):
    table = ddb.Table(TABLE_NAME)
    expr  = "SET " + ", ".join(f"#{k}=:{k}" for k in kwargs)
    names = {f"#{k}": k for k in kwargs}
    vals  = {f":{k}": v for k, v in kwargs.items()}
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=expr,
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=vals,
    )


def _ext(key: str) -> str:
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


# ─────────────────────────── RAW decode ─────────────────────────────────────

def decode_raw(file_bytes: bytes, params: dict) -> np.ndarray:
    """Demosaic a RAW file using rawpy. Returns H×W×3 uint8 array."""
    if not RAWPY_AVAILABLE:
        raise RuntimeError("rawpy not available in this environment")

    # Map our white-balance temperature shift to rawpy multipliers.
    # rawpy accepts (R, G, G, B) camera multipliers.
    temp_shift = params.get("temp", 0) / 100.0  # -1 … +1
    r_mult = 1.0 + temp_shift * 0.4
    b_mult = 1.0 - temp_shift * 0.4

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
        tf.write(file_bytes)
        tf.flush()
        with rawpy.imread(tf.name) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                user_wb=(r_mult, 1.0, 1.0, b_mult),
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                no_auto_bright=False,
                bright=max(0.1, 1.0 + params.get("exposure", 0) * 0.3),
            )
    return rgb  # numpy uint8 H×W×3


def decode_jpeg(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# ─────────────────────────── numpy edits ────────────────────────────────────

def apply_exposure(arr: np.ndarray, ev: float) -> np.ndarray:
    """Multiply by 2^ev, clamp to [0, 255]."""
    factor = 2.0 ** ev
    return np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_tone(arr: np.ndarray, highlights: float, shadows: float,
               whites: float, blacks: float) -> np.ndarray:
    """Tone-curve adjustments via per-channel LUT."""
    f32  = arr.astype(np.float32) / 255.0
    lut  = np.linspace(0, 1, 256, dtype=np.float32)

    # Highlights: compress/expand top quarter
    h = highlights / 100.0
    lut = np.where(lut > 0.75, lut + h * (1.0 - lut) * 0.5, lut)

    # Shadows: lift/crush bottom quarter
    s = shadows / 100.0
    lut = np.where(lut < 0.25, lut + s * lut * 0.5, lut)

    # Whites / Blacks
    lut = lut * (1.0 + whites / 100.0 * 0.5)
    lut = lut + blacks / 100.0 * 0.05
    lut = np.clip(lut, 0, 1)

    idx  = (f32 * 255).astype(np.uint8)
    out  = lut[idx]
    return (out * 255).astype(np.uint8)


def apply_white_balance(arr: np.ndarray, temp: float, tint: float) -> np.ndarray:
    """Simple per-channel WB multipliers derived from temp/tint sliders."""
    t = temp / 100.0   # -1 … +1 (negative = cooler / bluer)
    g = tint / 100.0   # -1 … +1

    r_mult = 1.0 + t * 0.35
    g_mult = 1.0 - abs(g) * 0.15
    b_mult = 1.0 - t * 0.35

    f32 = arr.astype(np.float32)
    f32[:, :, 0] = np.clip(f32[:, :, 0] * r_mult, 0, 255)
    f32[:, :, 1] = np.clip(f32[:, :, 1] * g_mult, 0, 255)
    f32[:, :, 2] = np.clip(f32[:, :, 2] * b_mult, 0, 255)
    return f32.astype(np.uint8)


def apply_saturation(arr: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
    """Adjust saturation/vibrance using HSL-inspired numpy ops."""
    f32 = arr.astype(np.float32) / 255.0
    r, g, b = f32[:, :, 0], f32[:, :, 1], f32[:, :, 2]

    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    lum   = (cmax + cmin) / 2.0
    chroma = cmax - cmin

    # Saturation scale
    s_scale = 1.0 + saturation / 100.0
    gray    = 0.2126 * r + 0.7152 * g + 0.0722 * b

    f32[:, :, 0] = np.clip(gray + (r - gray) * s_scale, 0, 1)
    f32[:, :, 1] = np.clip(gray + (g - gray) * s_scale, 0, 1)
    f32[:, :, 2] = np.clip(gray + (b - gray) * s_scale, 0, 1)

    # Vibrance: protect already-saturated colours
    if vibrance != 0:
        v_scale  = 1.0 + vibrance / 100.0
        sat_mask = 1.0 - chroma  # boost unsaturated more
        vf       = 1.0 + (v_scale - 1.0) * sat_mask
        gray2    = 0.2126 * f32[:,:,0] + 0.7152 * f32[:,:,1] + 0.0722 * f32[:,:,2]
        for c in range(3):
            f32[:, :, c] = np.clip(gray2 + (f32[:, :, c] - gray2) * vf, 0, 1)

    return (f32 * 255).astype(np.uint8)


def apply_sharpness(arr: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp-mask style sharpening via numpy convolution."""
    if amount <= 0:
        return arr
    # 3×3 Laplacian-based sharpening kernel
    strength = amount / 150.0
    kernel   = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0],
    ], dtype=np.float32) * strength

    from numpy.lib.stride_tricks import as_strided

    f32 = arr.astype(np.float32)
    out = f32.copy()
    # Simple 2D convolution (no scipy — pure numpy)
    padded = np.pad(f32, ((1, 1), (1, 1), (0, 0)), mode="edge")
    for dy in range(3):
        for dx in range(3):
            k = kernel[dy, dx]
            if k != 0:
                out += padded[dy:dy+arr.shape[0], dx:dx+arr.shape[1], :] * k
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise_reduction(arr: np.ndarray, amount: float) -> np.ndarray:
    """Box-blur noise reduction — pure numpy, no extra deps."""
    if amount <= 0:
        return arr
    radius = max(1, int(amount / 40))
    kernel_size = 2 * radius + 1
    f32 = arr.astype(np.float32)
    # Separable box blur: apply 1D average along rows then columns
    for _ in range(2):  # two-pass for box approximation
        f32 = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same"),
            0, f32
        )
        f32 = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same"),
            1, f32
        )
    # Blend with original based on amount
    alpha = amount / 100.0 * 0.6
    blended = (1 - alpha) * arr.astype(np.float32) + alpha * f32
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────── pipeline ───────────────────────────────────────

def process_image(file_bytes: bytes, filename: str, params: dict) -> bytes:
    ext = _ext(filename)

    # Decode
    if ext in RAW_EXTENSIONS:
        arr = decode_raw(file_bytes, params)
    else:
        arr = decode_jpeg(file_bytes)

    # JPEG path: apply exposure separately (rawpy handles it for RAW)
    if ext not in RAW_EXTENSIONS:
        arr = apply_exposure(arr, params.get("exposure", 0))

    # Apply white balance (fine-tuning on top of rawpy's demosaic)
    arr = apply_white_balance(arr, params.get("temp", 0), params.get("tint", 0))

    # Tone
    arr = apply_tone(arr,
        highlights=params.get("highlights", 0),
        shadows=params.get("shadows", 0),
        whites=params.get("whites", 0),
        blacks=params.get("blacks", 0),
    )

    # Colour
    arr = apply_saturation(arr,
        saturation=params.get("saturation", 0),
        vibrance=params.get("vibrance", 0),
    )

    # Detail
    arr = apply_sharpness(arr, params.get("sharpness", 0))
    arr = apply_noise_reduction(arr, params.get("nr", 0))

    # Encode to JPEG
    quality = int(params.get("quality", 90))
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ─────────────────────────── Lambda entry ───────────────────────────────────

def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return _cors({})

    try:
        body   = json.loads(event.get("body") or "{}")
        job_id = body["job_id"]
        params = body.get("params", {})
    except (KeyError, json.JSONDecodeError):
        return _cors({"error": "Missing job_id or invalid body"}, 400)

    # Fetch job from DynamoDB to get s3_key and filename
    table = ddb.Table(TABLE_NAME)
    try:
        item = table.get_item(Key={"job_id": job_id}).get("Item")
        if not item:
            return _cors({"error": "Job not found"}, 404)
    except ClientError as e:
        return _cors({"error": e.response["Error"]["Message"]}, 500)

    s3_key   = item["s3_key"]
    filename = item["filename"]

    _update_job(job_id, status="processing", started_at=int(time.time()))

    # Download from S3
    try:
        obj        = s3.get_object(Bucket=UPLOAD_BUCKET, Key=s3_key)
        file_bytes = obj["Body"].read()
    except ClientError as e:
        _update_job(job_id, status="failed", error=e.response["Error"]["Message"])
        return _cors({"error": f"S3 read error: {e.response['Error']['Message']}"}, 500)

    # Process
    try:
        output_bytes = process_image(file_bytes, filename, params)
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))
        return _cors({"error": f"Processing error: {str(e)}"}, 500)

    # Upload result
    result_key = f"processed/{job_id}/result.jpg"
    try:
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=result_key,
            Body=output_bytes,
            ContentType="image/jpeg",
            Metadata={"job_id": job_id, "source": filename},
        )
    except ClientError as e:
        _update_job(job_id, status="failed", error=e.response["Error"]["Message"])
        return _cors({"error": f"S3 write error: {e.response['Error']['Message']}"}, 500)

    _update_job(job_id,
        status="complete",
        result_key=result_key,
        completed_at=int(time.time()),
        output_size=len(output_bytes),
    )

    return _cors({
        "job_id":     job_id,
        "result_key": result_key,
        "size_bytes": len(output_bytes),
    })