from abc import ABC, abstractmethod
from typing import Any, Mapping


class DBInterface(ABC):
    """Abstract asynchronous database interface"""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database"""
        pass

    @abstractmethod
    async def query(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query"""
        pass
