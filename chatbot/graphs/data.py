from enum import StrEnum


class Source(StrEnum):
    """
    Enum representing different sources of documentation or knowledge.

    - `unimatica`: contains information about italian CCNL unimatica contract.
    """

    unimatica = "unimatica"


class DBName(StrEnum):
    """
    Enum representing different database names available.

    - `cameratraps`: contains information about wild life monitoring with cameratraps.
    """

    cameratraps = "cameratraps"
