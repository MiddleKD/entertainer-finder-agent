import json
import os
import random
from typing import Dict, Iterable, List, Literal, Union

import kagglehub
import numpy as np
import requests
from kagglehub import KaggleDatasetAdapter
from tqdm import tqdm

from src.db import VectorDBClient
from src.model import PromptEmbeddingModel


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
    def __init__(
        self,
        lfw_match_data_num: int,
        lfw_mismatch_data_num: int,
        face_data_num: int,
        retrieve_data_num: int,
        db_client: VectorDBClient,
        retrieve_query_path: str,
        retrieve_vector_path: str,
        n8n_url: str,
        seed: int = 42,
        text_model: PromptEmbeddingModel = None,
    ) -> None:

        random.seed(seed)

        lfw_root_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        self.lfw_root_path = os.path.join(
            lfw_root_path, "lfw-deepfunneled", "lfw-deepfunneled"
        )
        self.db_client = db_client
        self.retrieve_query_path = retrieve_query_path
        self.retrieve_vector_path = retrieve_vector_path
        self.n8n_url = n8n_url
        self.text_model = text_model

        # lfw match dataset
        self.lfw_match_dataset = list(
            kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "jessicali9530/lfw-dataset",
                "matchpairsDevTrain.csv",
            )
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)[:lfw_match_data_num]
            .T.to_dict()
            .values()
        )

        # lfw mismatch dataset
        self.lfw_mismatch_dataset = list(
            kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "jessicali9530/lfw-dataset",
                "mismatchpairsDevTrain.csv",
            )
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)[:lfw_mismatch_data_num]
            .T.to_dict()
            .values()
        )

        db_data_num = self.db_client.get_collection_info()["points_count"]
        points = self.db_client.fetch(
            random.choices(range(0, db_data_num), k=face_data_num + 100)
        )

        # face dataset
        self.face_dataset = [
            {"main_face": point.vector["main_face"], "face": point.vector["face"]}
            for point in points
        ][:face_data_num]

        # retrieve dataset
        meta_infos = [
            {
                "id": point.id,
                "meta": point.vector["prompt"],
                "prompt_source": point.payload["prompt_source"],
            }
            for point in points
        ][:retrieve_data_num]

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
                "prompt": vector_map[str(info["id"])],
            }
            for info in meta_infos
        ]

    def _lfw_match_preprocess(self, dataset: list):
        dataset = [
            {
                "name": data["name"],
                "imagenum1": os.path.join(
                    self.lfw_root_path,
                    data["name"],
                    f"{data['name']}_{str(data['imagenum1']).zfill(4)}.jpg",
                ),
                "imagenum2": os.path.join(
                    self.lfw_root_path,
                    data["name"],
                    f"{data['name']}_{str(data['imagenum2']).zfill(4)}.jpg",
                ),
            }
            for data in dataset
        ]
        return dataset

    def _lfw_mismatch_preprocess(self, dataset: list):
        dataset = [
            {
                "name": data["name"],
                "name.1": data["name.1"],
                "imagenum1": os.path.join(
                    self.lfw_root_path,
                    data["name"],
                    f"{data['name']}_{str(data['imagenum1']).zfill(4)}.jpg",
                ),
                "imagenum2": os.path.join(
                    self.lfw_root_path,
                    data["name.1"],
                    f"{data['name.1']}_{str(data['imagenum2']).zfill(4)}.jpg",
                ),
            }
            for data in dataset
        ]
        return dataset

    def _make_retrieve_query(self, dataset: Iterable):
        queries = {}
        for data in tqdm(
            dataset, total=len(dataset), desc="Make retrieve query by N8N"
        ):
            res = requests.post(self.n8n_url, data=data["prompt_source"])
            queries[data["id"]] = res.text
        with open(self.retrieve_query_path, mode="w") as f:
            json.dump(queries, f, indent=4, ensure_ascii=False)
        return queries

    def _make_retrieve_vector(self, query_map: Union[Dict[str, str], Dict[int, str]]):
        vectors = {}
        for id, query in tqdm(
            query_map.items(),
            total=len(query_map),
            desc="Make retrieve query embedding",
        ):
            embedding = self.text_model.embed(query)
            vectors[str(id)] = embedding
        np.savez(self.retrieve_vector_path, data=vectors)
        return vectors

    def get_data(
        self, type_of_dataset: Literal["lfw_match", "lfw_mismatch", "face", "retrieve"]
    ):
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
