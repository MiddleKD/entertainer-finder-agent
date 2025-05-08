import sys
from typing import Iterable

import numpy as np
from tqdm import tqdm

sys.path.append("src")
from dataset import DatasetManager

from src.constant import N8N_SUMMARIZE_PROMPT_URL, VECTOR_DB_COLLECTION, VECTOR_DB_URL
from src.db import VectorDBClient
from src.model import FaceEmbeddingModel, PromptEmbeddingModel

RANDOM_SEED = 42
N8N_URL = N8N_SUMMARIZE_PROMPT_URL
RETRIEVE_QUERY_PATH = "benchmark/dataset/retrieve_queries.json"
RETRIEVE_VECTOR_PATH = "benchmark/dataset/retrieve_vectors.npz"


def print_metric(title, value, dataset_size=None):
    print(f"\n--- {title} ---")
    if dataset_size:
        print(f"  → Dataset Size: {dataset_size}")
    print(f"  → Accuracy: \033[1m{value:.4f}\033[0m")


class Benchmark:
    def __init__(
        self, face_model: FaceEmbeddingModel, db_client: VectorDBClient
    ) -> None:
        self.face_model = face_model
        self.db_client = db_client

    def face_feature_extraction_accuracy(
        self, match_dataset: Iterable, mismatch_dataset: Iterable
    ):

        correct_match = 0
        correct_mismatch = 0
        total_num = len(match_dataset) + len(mismatch_dataset)

        for data in tqdm(
            match_dataset, total=total_num, initial=0, desc="Processing Match dataset"
        ):
            if self.face_model.is_same_face(data["imagenum1"], data["imagenum2"]):
                correct_match += 1

        for data in tqdm(
            mismatch_dataset,
            total=total_num,
            initial=len(match_dataset),
            desc="Processing Mismatch dataset",
        ):
            if not self.face_model.is_same_face(data["imagenum1"], data["imagenum2"]):
                correct_mismatch += 1

        return (correct_match + correct_mismatch) / total_num * 100

    def featuremap_generation_accuracy(self, dataset: Iterable):

        similarities = []
        for data in tqdm(dataset):
            main_face = np.array(data["main_face"])
            other_face = np.array(data["face"])
            similarities.append(
                (main_face @ other_face.T).mean()
            )  # Cuz all vectors are already normalized.

        return np.mean(similarities)

    def data_linkage_analize_accuracy(
        self, dataset: Iterable, k=3, a=0.05, b=0.4725, r=0.4725
    ):

        rtp = 0
        rtn = 0  # 정의되지 않음
        rfp = 0
        rfn = 0
        n = len(dataset)

        for data in tqdm(dataset):
            id = data["id"]
            main_face = data["main_face"]
            prompt = data["prompt"]

            points = self.db_client.query_multidomain(
                query_vectors_1=main_face,
                vector_domain_1="face",
                query_vectors_2=prompt,
                vector_domain_2="prompt",
                limit=k,
            )

            retrieved_point_ids = [point.id for point in points]

            if id in retrieved_point_ids:
                rtp += 1
            else:
                rfn += 1

            for retrieved_id in retrieved_point_ids:
                if retrieved_id != id:
                    rfp += 1

        precision = rtp / (rtp + rfp)
        recall = rtp / (rtp + rfn)
        accuracy = (rtp + rtn) / n

        return (a * precision + b * recall + r * accuracy) * 100


if __name__ == "__main__":
    face_model = FaceEmbeddingModel()
    text_model = PromptEmbeddingModel()
    db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)

    datasetmanager = DatasetManager(
        lfw_match_data_num=500,
        lfw_mismatch_data_num=500,
        face_data_num=1000,
        retrieve_data_num=200,
        db_client=db_client,
        retrieve_query_path=RETRIEVE_QUERY_PATH,
        retrieve_vector_path=RETRIEVE_VECTOR_PATH,
        n8n_url=N8N_SUMMARIZE_PROMPT_URL,
        seed=RANDOM_SEED,
        text_model=text_model,
    )
    benchmark = Benchmark(face_model=face_model, db_client=db_client)

    # face_feature_extraction_accuracy
    match_dataset = datasetmanager.get_data("lfw_match")
    mismatch_dataset = datasetmanager.get_data("lfw_mismatch")
    face_feature_extraction_accuracy = benchmark.face_feature_extraction_accuracy(
        match_dataset, mismatch_dataset
    )
    print_metric(
        "Face Feature Extraction Accuracy",
        face_feature_extraction_accuracy,
        dataset_size=len(match_dataset) + len(mismatch_dataset),
    )

    # featuremap_generation_accuracy
    face_dataset = datasetmanager.get_data("face")
    featuremap_generation_accuracy = benchmark.featuremap_generation_accuracy(
        face_dataset
    )
    print_metric(
        "Feature Map Generation Accuracy",
        featuremap_generation_accuracy,
        dataset_size=len(face_dataset),
    )

    # data_linkage_analize_accuracy
    retrieve_dataset = datasetmanager.get_data("retrieve")
    data_linkage_analize_accuracy = benchmark.data_linkage_analize_accuracy(
        retrieve_dataset, k=4, a=0.05, b=0.4725, r=0.4725
    )
    print_metric(
        "Data Linkage Analize Accuracy",
        data_linkage_analize_accuracy,
        dataset_size=len(retrieve_dataset),
    )
