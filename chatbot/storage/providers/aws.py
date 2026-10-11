import asyncio
import re

import boto3
from botocore.exceptions import ClientError

from chatbot.errors import StorageException
from chatbot.settings import S
from chatbot.storage.domain import BaseStorage


class S3Storage(BaseStorage):
    """S3 storage class"""

    def __init__(self, **kwargs):
        self.s3 = boto3.client("s3", **kwargs)

    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Upload file to S3 using the default bucket name"""

        def _upload():
            self.s3.put_object(Bucket=S.STORAGE_BUCKET, Key=key, Body=data, ContentType=content_type)

        await asyncio.to_thread(_upload)

        return f"s3://{S.STORAGE_BUCKET}/{key}"

    async def download(self, file_uri: str, size_limit: int = 0) -> bytes:
        """Download file from S3"""

        match = re.match(pattern=r"s3://([^/]+)/(.*)", string=file_uri)
        if not match:
            raise StorageException(f"Invalid S3 URI format: {file_uri}")

        bucket, key = match.groups()

        def _download() -> bytes:
            try:
                head = self.s3.head_object(Bucket=bucket, Key=key)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code in ("404", "NoSuchKey"):
                    raise StorageException(f"File not found: {file_uri}")
                raise StorageException(f"Error checking file {file_uri}: {e}")

            file_size_bytes = head.get("ContentLength", 0)
            size_kb = round(file_size_bytes / 1024, 2)
            if 0 < size_limit < size_kb:
                raise StorageException(f"File exceeds size limit: {size_kb:.2f} KB > {size_limit:.2f} KB")

            try:
                response = self.s3.get_object(Bucket=bucket, Key=key)
                return response["Body"].read()
            except ClientError as e:
                raise StorageException(f"Error downloading file {file_uri}: {e}")

        return await asyncio.to_thread(_download)
