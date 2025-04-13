from fastapi import File, Body, UploadFile, Form
from pydantic import BaseModel
from typing import Optional

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
