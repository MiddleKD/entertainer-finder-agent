import os, sys
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Iterable, Literal
from tqdm import tqdm
import random
import numpy as np
import requests
import json

sys.path.append("src")
from src.model import FaceEmbeddingModel
from src.constant import VECTOR_DB_URL, VECTOR_DB_COLLECTION
from src.db import VectorDBClient

RANDOM_SEED = 42
N8N_URL = "http://localhost:5678/webhook/e01a6079-9677-43f6-b1be-f5e25f864b0b"
RETRIEVE_QUERY_PATH = "benchmark/dataset/retrieve_queries.json"

class DatasetManager:
    def __init__(self, 
            lfw_match_data_num:int, 
            lfw_mismatch_data_num:int, 
            face_data_num:int, 
            retrieve_data_num:int, 
            retrieve_query_path:str = RETRIEVE_QUERY_PATH,
            n8n_url:str = N8N_URL
        ) -> None:

        random.seed(RANDOM_SEED)

        lfw_root_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")   
        self.lfw_root_path = os.path.join(lfw_root_path, "lfw-deepfunneled", "lfw-deepfunneled")
        self.retrieve_query_path = retrieve_query_path
        self.n8n_url = n8n_url


        # lfw match dataset
        self.lfw_match_dataset = list(kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "jessicali9530/lfw-dataset",
            "matchpairsDevTrain.csv"
        ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)[:lfw_match_data_num].T.to_dict().values())


        # lfw mismatch dataset
        self.lfw_mismatch_dataset = list(kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "jessicali9530/lfw-dataset",
            "mismatchpairsDevTrain.csv"
        ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)[:lfw_mismatch_data_num].T.to_dict().values())

        self.db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)
        db_data_num = self.db_client.get_collection_info()["points_count"]
        points = self.db_client.fetch(random.choices(range(0, db_data_num), k=face_data_num+100))
        

        # face dataset
        self.face_dataset = [{"main_face":point.vector["main_face"], "face":point.vector["face"]} for point in points][:face_data_num]


        # retrieve dataset        
        meta_infos = [{"id":point.id, "meta":point.vector["prompt"], "prompt_source":point.payload["prompt_source"]} for point in points][:retrieve_data_num]
        if not os.path.exists(self.retrieve_query_path):
            self._make_retrieve_query(meta_infos)
        with open(self.retrieve_query_path, mode="r") as f:
            query_map = json.load(f)
        self.retrieve_dataset = [{"query":query_map[str(info["id"])], **info} for info in meta_infos]

    def _lfw_match_preprocess(self, dataset:list):
        dataset = [
            {
                "name":data["name"], 
                "imagenum1": os.path.join(self.lfw_root_path, data['name'], f"{data['name']}_{str(data['imagenum1']).zfill(4)}.jpg"),
                "imagenum2": os.path.join(self.lfw_root_path, data['name'], f"{data['name']}_{str(data['imagenum2']).zfill(4)}.jpg")
            } 
            for data in dataset
        ]
        return dataset

    def _lfw_mismatch_preprocess(self, dataset:list):
        dataset = [
            {
                "name":data["name"],
                "name.1":data["name.1"],
                "imagenum1": os.path.join(self.lfw_root_path, data['name'], f"{data['name']}_{str(data['imagenum1']).zfill(4)}.jpg"),
                "imagenum2": os.path.join(self.lfw_root_path, data['name.1'], f"{data['name.1']}_{str(data['imagenum2']).zfill(4)}.jpg")
            } 
            for data in dataset
        ]
        return dataset

    def _make_retrieve_query(self, dataset:Iterable):
        queries = {}
        for data in tqdm(dataset, total=len(dataset), desc="Make retrieve query by N8N"):
            res = requests.post(self.n8n_url, data=data["prompt_source"])
            queries[data["id"]] = res.text
        with open(self.retrieve_query_path, mode="w") as f:
            json.dump(queries, f, indent=4, ensure_ascii=False)


    def get_data(self, type_of_dataset:Literal["lfw_match", "lfw_mismatch", "face", "retrieve"]):
        match type_of_dataset:
            case "lfw_match":
                return self._lfw_match_preprocess(self.lfw_match_dataset)
            case "lfw_mismatch":
                return self._lfw_mismatch_preprocess(self.lfw_mismatch_dataset)
            case "face":
                return self.face_dataset
            case "retrieve":
                return self.retrieve_dataset
        return None


class Benchmark:
    def __init__(self) -> None:
        self.face_model = FaceEmbeddingModel()

    def face_feature_extraction_accuracy(self, match_dataset:Iterable, mismatch_dataset:Iterable):

        correct_match = 0
        correct_mismatch = 0
        total_num = len(match_dataset) + len(mismatch_dataset)

        for data in tqdm(match_dataset, total=total_num, initial=0, desc="Processing Match dataset"):
            if self.face_model.is_same_face(data["imagenum1"], data["imagenum2"]):
                correct_match += 1

        for data in tqdm(mismatch_dataset, total=total_num, initial=len(match_dataset), desc="Processing Mismatch dataset"):
            if not self.face_model.is_same_face(data["imagenum1"], data["imagenum2"]):
                correct_mismatch += 1

        return (correct_match + correct_mismatch) / total_num * 100
        

    def featuremap_generation_accuracy(self, dataset:Iterable):
        
        similarities = []
        for data in tqdm(dataset):
            main_face = np.array(data["main_face"])
            other_face = np.array(data["face"])
            similarities.append((main_face @ other_face.T).mean()) # Cuz all vectors are already normalized.

        return np.mean(similarities)

    def data_linkage_analize_accuracy(self, dataset:Iterable):
        pass

    def matching_satisfaction(self, dataset:Iterable):
        return None


if __name__ == "__main__":
    datasetmanager = DatasetManager(
        lfw_match_data_num=500, 
        lfw_mismatch_data_num=500, 
        face_data_num=1000, 
        retrieve_data_num=200
    )
    benchmark = Benchmark()

    # match_dataset = datasetmanager.get_data("lfw_match")
    # mismatch_dataset = datasetmanager.get_data("lfw_mismatch")

    # face_feature_extraction_accuracy = benchmark.face_feature_extraction_accuracy(match_dataset, mismatch_dataset)
    # print(face_feature_extraction_accuracy)

    # face_dataset = datasetmanager.get_data("face")
    # featuremap_generation_accuracy = benchmark.featuremap_generation_accuracy(face_dataset)
    # print(featuremap_generation_accuracy)

    retrieve_dataset = datasetmanager.get_data("retrieve")
