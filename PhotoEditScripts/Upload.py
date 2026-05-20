"""
Lambda: upload
Validates the incoming file type and returns a presigned S3 PUT URL so the
browser can upload the raw file directly (bypassing Lambda's 10 MB payload cap).
Also creates the initial DynamoDB job record.
"""
import json
import os
import uuid
import time
import boto3
from botocore.exceptions import ClientError

s3  = boto3.client("s3")
ddb = boto3.resource("dynamodb")

UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]
TABLE_NAME    = os.environ["JOBS_TABLE"]
URL_EXPIRY    = 900  # 15 minutes

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg",
    "raw", "nef", "cr2", "cr3",
    "arw", "dng", "orf", "rw2",
}

CONTENT_TYPE_MAP = {
    "image/jpeg":       "jpeg",
    "image/x-nikon-nef":"nef",
    "image/x-raw":      "raw",
    "application/octet-stream": None,  # infer from filename
}


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


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return _cors({})

    try:
        body     = json.loads(event.get("body") or "{}")
        filename = body.get("filename", "").strip()
        ctype    = body.get("content_type", "application/octet-stream")
    except (json.JSONDecodeError, AttributeError):
        return _cors({"error": "Invalid request body"}, 400)

    ext = _ext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return _cors({"error": f"Unsupported file type: .{ext}"}, 415)

    job_id   = str(uuid.uuid4())
    s3_key   = f"uploads/{job_id}/{filename}"

    # ── Create DynamoDB job record ──────────────────────────────────────
    table = ddb.Table(TABLE_NAME)
    try:
        table.put_item(Item={
            "job_id":     job_id,
            "status":     "pending_upload",
            "filename":   filename,
            "s3_key":     s3_key,
            "created_at": int(time.time()),
            "ttl":        int(time.time()) + 86400,  # expire in 24 h
        })
    except ClientError as e:
        return _cors({"error": f"DB error: {e.response['Error']['Message']}"}, 500)

    # ── Generate presigned PUT URL ───────────────────────────────────────
    try:
        presigned = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket":      UPLOAD_BUCKET,
                "Key":         s3_key,
                "ContentType": ctype,
            },
            ExpiresIn=URL_EXPIRY,
        )
    except ClientError as e:
        return _cors({"error": f"S3 error: {e.response['Error']['Message']}"}, 500)

    return _cors({
        "job_id":     job_id,
        "upload_url": presigned,
        "s3_key":     s3_key,
        "expires_in": URL_EXPIRY,
    })