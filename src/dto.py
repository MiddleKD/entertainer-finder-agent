from typing import Optional

from fastapi import Body, File, Form, UploadFile
from pydantic import BaseModel


class ImageQuery(BaseModel):
    filename: str
    mimetype: str
    size: int


class EmbeddingResponse(BaseModel):
    item_id: int
    q: Optional[str] = None


class EmbeddingRequest(BaseModel):
    image_query: UploadFile = File(...)
    text_query: str
