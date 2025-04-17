import io
import os
import uuid

from fastapi import FastAPI
from PIL import Image

from dto import *
from constant import UPLOAD_CACHE_DIR

app = FastAPI()


@app.get("/")
async def healthcheck():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    image_data = await file.read()

    file_name = str(uuid.uuid4()) + ".jpg"
    Image.open(io.BytesIO(image_data)).convert("RGB").save(
        os.path.join(UPLOAD_CACHE_DIR, file_name)
    )
    return {"image_name": file_name}
