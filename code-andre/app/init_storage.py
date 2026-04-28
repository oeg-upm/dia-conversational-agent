from .storage import ensure_bucket
from .config import settings

def main():
    ensure_bucket(settings.S3_BUCKET_RAW)
    ensure_bucket(settings.S3_BUCKET_PARSED)
    ensure_bucket(settings.S3_BUCKET_LOGS)
    print("✅ Buckets ready:",
          settings.S3_BUCKET_RAW,
          settings.S3_BUCKET_PARSED,
          settings.S3_BUCKET_LOGS)

if __name__ == "__main__":
    main()