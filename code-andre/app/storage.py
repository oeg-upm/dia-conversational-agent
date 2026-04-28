import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from .config import settings
from typing import Optional

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket(bucket: str):
    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)

def put_bytes(bucket: str, key: str, data: bytes, content_type: Optional[str] = None):
    s3 = s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3.put_object(Bucket=bucket, Key=key, Body=data, **extra)

def get_bytes(bucket: str, key: str) -> bytes:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def exists(bucket: str, key: str) -> bool:
    s3 = s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False