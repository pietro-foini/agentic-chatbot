from chatbot.settings import CloudProvider, S
from chatbot.storage.domain import BaseStorage
from chatbot.storage.providers import GCSStorage, S3Storage


def get_storage_client() -> BaseStorage:
    """Load the storage client"""

    if S.CLOUD_PROVIDER == CloudProvider.AWS:
        return S3Storage()
    elif S.CLOUD_PROVIDER == CloudProvider.GCP:
        return GCSStorage()
    else:
        raise Exception("Invalid provider")


storage_client: BaseStorage = get_storage_client()
