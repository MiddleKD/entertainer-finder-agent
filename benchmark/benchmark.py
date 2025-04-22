import os, sys
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Iterable, Literal, Union, List, Dict
from tqdm import tqdm
import random
import numpy as np
import requests
import json

sys.path.append("src")
from src.model import FaceEmbeddingModel, PromptEmbeddingModel
from src.constant import VECTOR_DB_URL, VECTOR_DB_COLLECTION, N8N_SUMMARIZE_PROMPT_URL
from src.db import VectorDBClient

RANDOM_SEED = 42
N8N_URL = N8N_SUMMARIZE_PROMPT_URL
RETRIEVE_QUERY_PATH = "benchmark/dataset/retrieve_queries.json"
RETRIEVE_VECTOR_PATH = "benchmark/dataset/retrieve_vectors.npz"

def remove_file(path: Union[str, List[str]]):
    if isinstance(path, str):
        path = [path]
    for p in path:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

def open_file(path: str):
    if path.endswith(".json"):
        with open(path, mode="r") as f:
            return json.load(f)
    elif path.endswith(".npz"):
        return np.load(path, allow_pickle=True)["data"].item()
    else:
        raise NameError("unknow file extension")

class DatasetManager:
    def __init__(self, 
            lfw_match_data_num:int, 
            lfw_mismatch_data_num:int, 
            face_data_num:int, 
            retrieve_data_num:int,
            db_client:VectorDBClient, 
            retrieve_query_path:str = RETRIEVE_QUERY_PATH,
            retrieve_vector_path:str = RETRIEVE_VECTOR_PATH,
            n8n_url:str = N8N_URL,
            text_model:PromptEmbeddingModel = None,
        ) -> None:

        random.seed(RANDOM_SEED)

        lfw_root_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")   
        self.lfw_root_path = os.path.join(lfw_root_path, "lfw-deepfunneled", "lfw-deepfunneled")
        self.db_client = db_client
        self.retrieve_query_path = retrieve_query_path
        self.retrieve_vector_path = retrieve_vector_path
        self.n8n_url = n8n_url
        self.text_model = text_model

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

        db_data_num = self.db_client.get_collection_info()["points_count"]
        points = self.db_client.fetch(random.choices(range(0, db_data_num), k=face_data_num+100))
        

        # face dataset
        self.face_dataset = [{"main_face":point.vector["main_face"], "face":point.vector["face"]} for point in points][:face_data_num]


        # retrieve dataset        
        meta_infos = [{"id":point.id, "meta":point.vector["prompt"], "prompt_source":point.payload["prompt_source"]} for point in points][:retrieve_data_num]

        is_query_exist = os.path.exists(self.retrieve_query_path)
        is_vector_exist = os.path.exists(self.retrieve_vector_path)
        if is_query_exist and is_vector_exist:
            query_map = open_file(self.retrieve_query_path)
            vector_map = open_file(self.retrieve_vector_path)
        elif is_query_exist and not is_vector_exist:
            query_map = open_file(self.retrieve_query_path)
            vector_map = self._make_retrieve_vector(query_map)
        else:
            remove_file([self.retrieve_query_path, self.retrieve_vector_path])
            query_map = self._make_retrieve_query(meta_infos)
            vector_map = self._make_retrieve_vector(query_map)
        
        self.retrieve_dataset = [
            {
                "id": info["id"],
                "main_face": self.db_client.fetch([info["id"]])[0].vector["main_face"],
                "prompt": vector_map[str(info["id"])]
            } for info in meta_infos
        ]


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
        return queries

    def _make_retrieve_vector(self, query_map:Union[Dict[str, str], Dict[int, str]]):
        vectors = {}
        for id, query in tqdm(query_map.items(), total=len(query_map), desc="Make retrieve query embedding"):
            embedding = self.text_model.embed(query)
            vectors[str(id)] = embedding
        np.savez(self.retrieve_vector_path, data=vectors)
        return vectors

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
    def __init__(self, face_model: FaceEmbeddingModel, db_client: VectorDBClient) -> None:
        self.face_model = face_model
        self.db_client = db_client

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

    def data_linkage_analize_accuracy(self, dataset:Iterable, k=3, a=0.05, b=0.4725, r=0.4725):
        
        rtp = 0
        rtn = 0 # 정의되지 않음
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
                limit=k
            )

            retrieved_point_ids = [point.id for point in points]

            if id in retrieved_point_ids:
                rtp += 1
            else:
                rfn += 1
            
            for retrieved_id in retrieved_point_ids:
                if retrieved_id != id:
                    rfp += 1

        precision = rtp/(rtp+rfp)
        recall = rtp/(rtp+rfn)
        accuracy = (rtp+rtn)/n

        return a*precision + b*recall + r*accuracy


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
        text_model=text_model
    )
    benchmark = Benchmark(
        face_model=face_model,
        db_client=db_client
    )


    # calculate face_feature_extraction_accuracy
    match_dataset = datasetmanager.get_data("lfw_match")
    mismatch_dataset = datasetmanager.get_data("lfw_mismatch")
    face_feature_extraction_accuracy = benchmark.face_feature_extraction_accuracy(match_dataset, mismatch_dataset)
    print(f"face_feature_extraction_accuracy: {face_feature_extraction_accuracy}")


    # calculate featuremap_generation_accuracy
    face_dataset = datasetmanager.get_data("face")
    featuremap_generation_accuracy = benchmark.featuremap_generation_accuracy(face_dataset)
    print(f"featuremap_generation_accuracy: {featuremap_generation_accuracy}")


    # calculate data_linkage_analize_accuracy
    retrieve_dataset = datasetmanager.get_data("retrieve")
    data_linkage_analize_accuracy = benchmark.data_linkage_analize_accuracy(retrieve_dataset, k=4)
    print(f"data_linkage_analize_accuracy: {data_linkage_analize_accuracy}")
