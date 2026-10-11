import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from chatbot.errors import UserFacingError
from chatbot.schemas.chat import Attachment
from chatbot.settings import S
from chatbot.storage.config import storage_client


async def save_attachment(file: UploadFile | None) -> Attachment | None:
    """Validate and upload the attachment to GCS, if there is one"""

    if not (file and file.filename and file.content_type and file.size):
        return None

    size_kb = round(file.size / 1024, 2)
    if file.content_type not in S.ALLOWED_MIME_TYPES:
        raise UserFacingError("File type not allowed.")
    if size_kb > S.MAX_ATTACHMENTS_SIZE:
        raise UserFacingError(f"The maximum allowed file size is {S.MAX_ATTACHMENTS_SIZE}KB.")

    content_bytes = await file.read()
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)
    file_id = f"{uuid.uuid4()}_{safe_name}"

    file_uri = await storage_client.upload(
        key=f"attachments/{file_id}", data=content_bytes, content_type=file.content_type
    )

    return Attachment(id=file_id, file_uri=file_uri, type=file.content_type, size=f"{size_kb}KB")
