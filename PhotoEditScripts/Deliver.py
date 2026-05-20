"""
Lambda: deliver
Looks up a completed job and returns a short-lived signed S3 URL
so the browser can download the processed image.
"""
import json
import os
import boto3
from botocore.exceptions import ClientError

s3  = boto3.client("s3")
ddb = boto3.resource("dynamodb")

OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
TABLE_NAME    = os.environ["JOBS_TABLE"]
URL_EXPIRY    = 3600  # 1 hour


def _cors(body, status=200):
    return {
        "statusCode": status,
        "headers": {
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET,OPTIONS",
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return _cors({})

    # result_key can come from path param or query string
    path_params   = event.get("pathParameters") or {}
    query_params  = event.get("queryStringParameters") or {}
    result_key    = path_params.get("result_key") or query_params.get("result_key")

    if not result_key:
        return _cors({"error": "result_key is required"}, 400)

    # Verify the key actually exists in S3
    try:
        s3.head_object(Bucket=OUTPUT_BUCKET, Key=result_key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return _cors({"error": "Result not found — processing may still be in progress"}, 404)
        return _cors({"error": f"S3 error: {e.response['Error']['Message']}"}, 500)

    # Generate signed GET URL
    try:
        download_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": OUTPUT_BUCKET,
                "Key":    result_key,
                "ResponseContentDisposition": "attachment; filename=edited_photo.jpg",
                "ResponseContentType": "image/jpeg",
            },
            ExpiresIn=URL_EXPIRY,
        )
    except ClientError as e:
        return _cors({"error": f"Could not generate URL: {e.response['Error']['Message']}"}, 500)

    return _cors({
        "download_url": download_url,
        "result_key":   result_key,
        "expires_in":   URL_EXPIRY,
    })