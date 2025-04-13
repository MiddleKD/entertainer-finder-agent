from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from constant import VECTOR_DB_URL, VECTOR_DB_COLLECTION

class VectorDBClient:
    def __init__(self, vectordb_url:str, collection_name:str):
        self.client = QdrantClient(vectordb_url)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    
    def insert():
    
    def insert_bulk():
    
    def delete():
    
    def query():
    

if __name__=="__main__":
    db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)
