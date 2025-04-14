import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from constant import CRAWL_DATA_PATH, VECTOR_DB_COLLECTION, VECTOR_DB_URL
from db import VectorDBClient
from model import FaceEmbeddingModel
from preprocess import Preprocessor


class VectorDBLoader:
    def __init__(self):
        self.db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)
        self.face_model = FaceEmbeddingModel()

    def _calculate_average_vector(self, vectors: List[List[float]]) -> List[float]:
        """
        여러 벡터의 평균을 계산합니다.

        Args:
            vectors: 평균을 계산할 벡터 리스트

        Returns:
            List[float]: 평균 벡터
        """
        if not vectors:
            raise ValueError("No vectors provided")

        # numpy 배열로 변환
        vectors_array = np.array(vectors)

        # 평균 계산
        average_vector = np.mean(vectors_array, axis=0)

        # 정규화 (선택적)
        average_vector = average_vector / np.linalg.norm(average_vector)

        return average_vector.tolist()

    def search_similar_faces_with_average(
        self,
        query_image_paths: Union[str, List[str]],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        여러 이미지의 얼굴 임베딩 평균을 사용하여 유사한 얼굴을 검색합니다.

        Args:
            query_image_paths: 검색할 이미지 경로 또는 경로 리스트
            limit: 반환할 결과 수
            score_threshold: 유사도 점수 임계값

        Returns:
            List[Dict]: 검색 결과 리스트
        """
        try:
            # 단일 이미지 경로를 리스트로 변환
            if isinstance(query_image_paths, str):
                query_image_paths = [query_image_paths]

            # 모든 이미지에서 얼굴 임베딩 추출
            all_embeddings = []
            for image_path in query_image_paths:
                embeddings = self.face_model.run_model(
                    image_path, only_return_face=True
                )
                if embeddings:
                    all_embeddings.extend(embeddings)

            if not all_embeddings:
                print("No faces detected in any of the query images")
                return []

            # 평균 벡터 계산
            average_vector = self._calculate_average_vector(all_embeddings)

            # 평균 벡터로 검색
            results = self.db_client.query(
                query_vector=average_vector,
                limit=limit,
                score_threshold=score_threshold,
            )

            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "score": result.score,
                        "member_id": result.payload["member_id"],
                        "name": result.payload["name"],
                        "prompt_source": result.payload["prompt_source"],
                        "image_index": result.payload["image_index"],
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching similar faces with average: {str(e)}")
            return []

    def load_data(self, csv_path: str):
        """
        CSV 파일의 데이터를 전처리하고 벡터DB에 로드합니다.

        Args:
            csv_path: 전처리할 CSV 파일 경로
        """
        # 데이터 전처리
        preprocessor = Preprocessor(csv_path)
        preprocessor.clean_na()
        processed_data_list = preprocessor.process_to_data_list()

        # 벡터DB에 데이터 로드
        for data in tqdm(processed_data_list, desc="Loading data to VectorDB"):
            try:
                # 프로필 이미지 경로 리스트 생성
                profile_images = [
                    os.path.join(CRAWL_DATA_PATH, img)
                    for img in data["payload"]["images"]
                ]

                # 얼굴 임베딩 추출
                face_embeddings = self.face_model.run_model(
                    profile_images, only_return_face=True
                )

                # 각 얼굴 임베딩을 별도의 포인트로 저장
                for idx, embedding in enumerate(face_embeddings):
                    point_id = f"{data['id']}_{idx}"  # 고유한 ID 생성

                    # 멀티벡터 구성
                    vectors = {
                        "face": embedding,  # 얼굴 임베딩
                        "prompt": data["vector"]["prompt"],  # 프롬프트 임베딩
                    }

                    # payload 구성
                    payload = {
                        "member_id": data["id"],
                        "name": data["payload"]["name"],
                        "prompt_source": data["payload"]["prompt_source"],
                        "image_index": idx,
                        "original_images": data["payload"]["images"],
                    }

                    # 벡터DB에 저장
                    self.db_client.insert(
                        point_id=point_id, vectors=vectors, payload=payload
                    )

                print(f"Successfully loaded data for member {data['id']}")

            except Exception as e:
                print(f"Error processing member {data['id']}: {str(e)}")

    def search_similar(
        self,
        query_face_image: Optional[str] = None,
        query_prompt: Optional[List[float]] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        얼굴 이미지와/또는 프롬프트를 사용하여 유사한 포인트를 검색합니다.

        Args:
            query_face_image: 검색할 얼굴 이미지 경로
            query_prompt: 검색할 프롬프트 임베딩
            limit: 반환할 결과 수
            score_threshold: 유사도 점수 임계값

        Returns:
            List[Dict]: 검색 결과 리스트
        """
        try:
            query_vectors = {}

            # 얼굴 이미지가 제공된 경우
            if query_face_image:
                face_embeddings = self.face_model.run_model(
                    query_face_image, only_return_face=True
                )
                if face_embeddings:
                    query_vectors["face"] = face_embeddings[0]

            # 프롬프트가 제공된 경우
            if query_prompt:
                query_vectors["prompt"] = query_prompt

            if not query_vectors:
                print("No query vectors provided")
                return []

            # 멀티벡터 검색
            results = self.db_client.query(
                query_vectors=query_vectors,
                limit=limit,
                score_threshold=score_threshold,
            )

            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "score": result.score,
                        "member_id": result.payload["member_id"],
                        "name": result.payload["name"],
                        "prompt_source": result.payload["prompt_source"],
                        "image_index": result.payload["image_index"],
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching similar points: {str(e)}")
            return []


if __name__ == "__main__":
    # 사용 예시
    loader = VectorDBLoader()

    # 데이터 로드
    loader.load_data("datas/member_202503311252.csv")

    # 여러 이미지의 평균을 사용한 유사 얼굴 검색 예시
    query_images = [
        "path/to/query/image1.jpg",
        "path/to/query/image2.jpg",
        "path/to/query/image3.jpg",
    ]

    results = loader.search_similar_faces_with_average(
        query_images, limit=5, score_threshold=0.7
    )

    for result in results:
        print(f"Similarity: {result['score']:.4f}")
        print(f"Member ID: {result['member_id']}")
        print(f"Name: {result['name']}")
        print(f"Prompt: {result['prompt_source']}")
        print("---")
