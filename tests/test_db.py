import numpy as np
import pytest

from constant import VECTOR_DB_COLLECTION, VECTOR_DB_URL
from db import VectorDBClient


@pytest.fixture
def db_client():
    """테스트용 VectorDBClient 인스턴스를 생성합니다."""
    client = VectorDBClient(VECTOR_DB_URL, "test_collection")
    yield client
    # 테스트 후 정리
    try:
        client.client.delete_collection("test_collection")
    except:
        pass


def test_initialization(db_client):
    """VectorDBClient 초기화 테스트"""
    assert db_client.collection_name == "test_collection"
    collection_info = db_client.get_collection_info()
    assert collection_info["config"]["params"]["vectors"]["face"]["size"] == 512
    assert collection_info["config"]["params"]["vectors"]["prompt"]["size"] == 1536
    assert (
        collection_info["config"]["params"]["vectors"]["face"]["distance"] == "Cosine"
    )
    assert (
        collection_info["config"]["params"]["vectors"]["prompt"]["distance"] == "Cosine"
    )


def test_insert_and_query(db_client):
    """단일 포인트 삽입 및 검색 테스트"""
    # 테스트 데이터
    test_id = 1
    face_vector = np.random.rand(512).tolist()
    prompt_vector = np.random.rand(1536).tolist()
    test_vector = {"face": face_vector, "prompt": prompt_vector, "main_face": face_vector}
    test_payload = {"name": "test", "value": 123}

    # 삽입
    assert db_client.insert(test_id, test_vector, test_payload)

    # 검색
    results = db_client.query(face_vector, limit=1, vector_domain="face")
    assert len(results) == 1
    assert results[0].id == test_id
    assert results[0].payload == test_payload
    assert results[0].score > 0.9  # 동일한 벡터이므로 높은 유사도 점수


def test_insert_bulk(db_client):
    """일괄 삽입 테스트"""
    # 테스트 데이터
    points = [
        {
            "id": i,
            "vectors": {
                "face": np.random.rand(512).tolist(),
                "prompt": np.random.rand(1536).tolist(),
                "main_face": np.random.rand(512).tolist(),
            },
            "payload": {"name": f"test_{i}", "value": i},
        }
        for i in range(5)
    ]

    # 일괄 삽입
    assert db_client.insert_bulk(points)

    # 각 포인트 검색
    for point in points:
        results = db_client.query(point["vectors"]["face"], limit=1, vector_domain="face")
        assert len(results) == 1
        assert results[0].id == point["id"]
        assert results[0].payload == point["payload"]


def test_delete(db_client):
    """삭제 테스트"""
    # 테스트 데이터 삽입
    test_id = 1
    face_vector = np.random.rand(512).tolist()
    prompt_vector = np.random.rand(1536).tolist()
    test_vector = {"face": face_vector, "prompt": prompt_vector, "main_face": face_vector}
    db_client.insert(test_id, test_vector)

    # 삭제
    assert db_client.delete(test_id)

    # 삭제 확인
    results = db_client.query(test_vector["face"], limit=1, vector_domain="face")
    assert len(results) == 0 or results[0].id != test_id


def test_delete_bulk(db_client):
    """일괄 삭제 테스트"""
    # 테스트 데이터 삽입
    point_ids = [1, 2, 3]
    for point_id in point_ids:
        face_vector = np.random.rand(512).tolist()
        prompt_vector = np.random.rand(1536).tolist()
        test_vector = {"face": face_vector, "prompt": prompt_vector, "main_face": face_vector}
        db_client.insert(point_id, test_vector)

    # 일괄 삭제
    assert db_client.delete(point_ids)

    # 삭제 확인
    for point_id in point_ids:
        results = db_client.query(np.random.rand(512).tolist(), limit=10, vector_domain="face")
        assert all(result.id != point_id for result in results)


def test_score_threshold(db_client):
    """유사도 점수 임계값 테스트"""
    # 테스트 데이터 삽입
    test_id = 1
    face_vector = np.random.rand(512).tolist()
    prompt_vector = np.random.rand(1536).tolist()
    test_vector = {"face": face_vector, "prompt": prompt_vector, "main_face": face_vector}
    db_client.insert(test_id, test_vector)

    # 다른 벡터로 검색 (낮은 유사도)
    different_vector = np.random.rand(512).tolist()
    results = db_client.query(different_vector, score_threshold=0.9, vector_domain="face")
    assert len(results) == 0

def test_collection_info(db_client):
    """컬렉션 정보 조회 테스트"""
    info = db_client.get_collection_info()
    assert "config" in info
    assert "status" in info
    assert "optimizer_status" in info
    assert "vectors_count" in info
