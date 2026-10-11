import asyncio
import re

from google.cloud import storage

from chatbot.errors import StorageException
from chatbot.settings import S
from chatbot.storage.domain import BaseStorage


class GCSStorage(BaseStorage):
    """GCS storage class"""

    def __init__(self):
        self.client = storage.Client()

    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Upload file to GCS using the default bucket name"""

        bucket = self.client.bucket(S.STORAGE_BUCKET)
        blob = bucket.blob(key)

        await asyncio.to_thread(blob.upload_from_string, data, content_type=content_type)

        return f"gs://{S.STORAGE_BUCKET}/{key}"

    async def download(self, file_uri: str, size_limit: int = 0) -> bytes:
        """Download file from GCS"""

        match = re.match(pattern=r"gs://([^/]+)/(.*)", string=file_uri)
        if not match:
            raise StorageException(f"Invalid GCS URI format: {file_uri}")

        bucket_name, blob_path = match.groups()
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Check if the file exists.
        exists = await asyncio.to_thread(blob.exists)
        if not exists:
            raise StorageException(f"File not found: {file_uri}")

        # Check the file size.
        await asyncio.to_thread(blob.reload)
        size_kb = round(blob.size / 1024, 2)
        if size_kb > size_limit:
            raise StorageException(f"File exceeds size limit: {size_kb:.2f} KB > {size_limit:.2f} KB")

        return await asyncio.to_thread(blob.download_as_bytes)
