import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from constant import VECTOR_DB_COLLECTION, VECTOR_DB_URL


class VectorDBClient:
    def __init__(self, vectordb_url: str, collection_name: str):
        """
        Qdrant 벡터 데이터베이스 클라이언트를 초기화합니다.

        Args:
            vectordb_url: Qdrant 서버 URL
            collection_name: 컬렉션 이름
        """
        try:
            self.client = QdrantClient(vectordb_url)
            self.collection_name = collection_name

            # 컬렉션이 존재하지 않으면 생성
            collections = self.client.get_collections().collections
            if not any(
                collection.name == collection_name for collection in collections
            ):
                # 멀티벡터 설정
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "face": VectorParams(
                            size=512,
                            distance=Distance.COSINE,
                            multivector_config=MultiVectorConfig(
                                comparator=MultiVectorComparator("max_sim")
                            ),
                        ),
                        "prompt": VectorParams(size=1536, distance=Distance.COSINE),
                        "main_face": VectorParams(size=512, distance=Distance.COSINE),
                    },
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VectorDBClient: {str(e)}")

    def insert(
        self,
        point_id: int,
        vectors: Dict[str, List[float]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        멀티벡터를 포함한 포인트를 벡터 데이터베이스에 삽입합니다.

        Args:
            point_id: 포인트 ID
            vectors: 벡터 이름과 벡터 데이터의 딕셔너리
            payload: 추가 메타데이터

        Returns:
            bool: 삽입 성공 여부
        """
        try:
            point = PointStruct(id=point_id, vector=vectors, payload=payload or {})
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to insert point: {str(e)}")

    def insert_bulk(self, points: List[Dict[str, Any]]) -> bool:
        """
        여러 멀티벡터 포인트를 벡터 데이터베이스에 일괄 삽입합니다.

        Args:
            points: 삽입할 포인트 리스트 (각 포인트는 id, vectors, payload를 포함)

        Returns:
            bool: 삽입 성공 여부
        """
        try:
            point_structs = [
                PointStruct(
                    id=point["id"],
                    vector=point["vectors"],
                    payload=point.get("payload", {}),
                )
                for point in points
            ]
            self.client.upsert(
                collection_name=self.collection_name, points=point_structs
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to insert bulk points: {str(e)}")

    def delete(self, point_id: Union[int, List[int]]) -> bool:
        """
        포인트를 벡터 데이터베이스에서 삭제합니다.

        Args:
            point_id: 삭제할 포인트 ID 또는 ID 리스트

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            point_ids = [point_id] if isinstance(point_id, int) else point_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete points: {str(e)}")

    def query(
        self,        
        query_vectors: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        vector_domain: Literal["face", "prompt", "main_face"] = "face",
        with_vectors: bool = False
    ) -> List[ScoredPoint]:
        """
        멀티벡터를 사용하여 유사한 포인트를 검색합니다.

        Args:
            query_vectors: 검색할 벡터 이름과 벡터 데이터의 딕셔너리
            limit: 반환할 결과 수
            score_threshold: 유사도 점수 임계값

        Returns:
            List[ScoredPoint]: 검색 결과 리스트
        """

        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vectors,
                using=vector_domain,
                with_vectors = with_vectors,
                limit=limit,
                score_threshold=score_threshold,
            ).points
        except Exception as e:
            raise RuntimeError(f"Failed to query points: {str(e)}")
        
    def query_multidomain(
        self,
        query_vectors_1: List[float],
        vector_domain_1: Literal["face", "prompt", "main_face"],
        query_vectors_2: List[float],
        vector_domain_2: Literal["face", "prompt", "main_face"],
        limit: int = 10,
    ) -> List[ScoredPoint]:
        """
        멀티벡터를 사용하여 유사한 포인트를 검색합니다.

        Args:
            query_vectors: 검색할 벡터 이름과 벡터 데이터의 딕셔너리
            limit: 반환할 결과 수
            score_threshold: 유사도 점수 임계값

        Returns:
            List[ScoredPoint]: 검색 결과 리스트
        """

        try:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vectors_1,
                using=vector_domain_1,
                with_vectors = True,
                limit=int(limit*1.2),
            ).points

            matrix = []
            for point in points:
                matrix.append(point.vector[vector_domain_2])
            
            matrix = np.array(matrix)
            top_k_idx = np.argsort(query_vectors_2 @ matrix.T)[-limit:][::-1]

            return [points[idx] for idx in top_k_idx]

        except Exception as e:
            raise RuntimeError(f"Failed to query points: {str(e)}")

    def fetch(self, point_ids:List[int]) -> List[Dict[str, Any]]:
        """
        컬렉션에서 지정된 포인트를 가져옵니다.

        Args:
            point_ids: 포인트 아이디 리스트

        Returns:
            List[Dict[str, Any]]: 반환된 포인트 리스트
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_vectors=True,
                with_payload=True,
            )
            for point in points:
                trimmed = point.vector["face"][:3]
                point.vector["face"] = trimmed
            return points
        except Exception as e:
            raise RuntimeError(f"Failed to fetch points: {str(e)}")
        
    def get_collection_info(self) -> Dict[str, Any]:
        """
        현재 컬렉션의 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 컬렉션 정보
        """
        try:
            return self.client.get_collection(self.collection_name).dict()
        except Exception as e:
            raise RuntimeError(f"Failed to get collection info: {str(e)}")


if __name__ == "__main__":
    try:
        db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)
        print("VectorDBClient initialized successfully")
        print("Collection info:", db_client.get_collection_info())
    except Exception as e:
        print(f"Error initializing VectorDBClient: {str(e)}")
