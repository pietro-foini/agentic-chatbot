from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """Base storage class"""

    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str) -> str:

        pass

    @abstractmethod
    async def download(self, file_uri: str, size_limit: int = 0) -> bytes:

        pass
