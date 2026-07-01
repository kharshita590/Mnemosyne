from __future__ import annotations

import json

import boto3
from botocore.exceptions import ClientError

from config.settings import settings

_client = None


def get_s3():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
    return _client


async def archive_raw_content(
    user_id: str,
    conversation_id: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    """Store raw conversation content in S3 for audit/reprocessing. Returns S3 key."""
    key = f"raw/{user_id}/{conversation_id}.json"
    payload = {"content": content, "metadata": metadata or {}}
    try:
        get_s3().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=json.dumps(payload),
            ContentType="application/json",
        )
    except ClientError:
        # S3 archival is best-effort — don't fail ingestion if S3 is unavailable
        pass
    return key
