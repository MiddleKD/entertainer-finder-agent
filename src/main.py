import io

from fastapi import FastAPI

from dto import *
from model import run_model

app = FastAPI()


@app.get("/")
async def healthcheck():
    return {"ok": True}


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    image_data = await file.read()
    img_embed = run_model(io.BytesIO(image_data))

    return {"message": f"{img_embed}"}


@app.post("/search")
async def search(text: str = Form(...), file: UploadFile = File(...)):
    image_data = await file.read()
    img_embed = run_model(io.BytesIO(image_data))

    return {"message": f"{img_embed}"}
