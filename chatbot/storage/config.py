from chatbot.settings import CloudProvider, S
from chatbot.storage.domain import BaseStorage


def get_storage_client() -> BaseStorage:
    """Load the storage client"""

    if S.CLOUD_PROVIDER == CloudProvider.AWS:
        from chatbot.storage.providers.aws import S3Storage

        return S3Storage()
    elif S.CLOUD_PROVIDER == CloudProvider.GCP:
        from chatbot.storage.providers.gcp import GCSStorage

        return GCSStorage()
    else:
        from chatbot.storage.providers.aws import S3Storage

        return S3Storage(
            endpoint_url=S.STORAGE_ENDPOINT_URL,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )


storage_client: BaseStorage = get_storage_client()
