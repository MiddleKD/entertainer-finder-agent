from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import numpy as np
from textwrap import dedent
import os

from db import VectorDBClient
from model import FaceEmbeddingModel
from constant import UPLOAD_CACHE_DIR, VECTOR_DB_URL, VECTOR_DB_COLLECTION

load_dotenv()

face_model = FaceEmbeddingModel()
db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)

mcp = FastMCP(
    name="Entertainer Search",
    version="0.0.1",
    description="Retrieve entertainer face embedding DB",
    port=8001,
)

@mcp.tool(
    name="get_lookalike_profile",
    description="Given a list of image paths, returns similar face profiles based on averaged face embeddings."
)
def get_lookalike_profile(image_paths: list[str]) -> str:
    image_paths = [os.path.join(UPLOAD_CACHE_DIR, path) for path in image_paths]
    face_embeddings = face_model.run_model(image_paths, only_return_face=True)
    points = db_client.query(
        query_vectors=np.array(face_embeddings).mean(axis=0),
        limit=50,
        score_threshold=0.8,
        vector_domain="face"
    )

    result_context = ""

    for point in points:
        result_context += dedent(f"""
            ---------------
            ID: {point.id}
            Name: {point.payload["name_kr"]}
            Birthday: {point.payload["birth"]}
            Image: {point.payload["image_url"]}
            Info: {point.payload["prompt_source"]}
            ---------------
        """)

    return result_context

if __name__ == "__main__":
    mcp.run(transport='sse')
