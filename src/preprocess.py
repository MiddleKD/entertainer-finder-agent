import os
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

from constant import CRAWL_BASE_URL, CRAWL_DATA_PATH, VECTOR_DB_URL, VECTOR_DB_COLLECTION
from model import FaceEmbeddingModel
from db import VectorDBClient


def validate(input_value):
    return_value = None
    if isinstance(input_value, str):
        if input_value == "None":
            pass
        elif len(input_value) == 0:
            pass
        else:
            return_value = input_value.replace("None", "").replace("null", "").strip()
    return return_value


def convert_timestamp_to_date(timestamp):
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def log(message):
    with open("error.log", "a") as f:
        f.write(f"{message}\n")


class Crawler:
    def __init__(self, base_url: str, base_save_path: str):
        self.base_url = base_url
        self.base_save_path = base_save_path

    def run(self, url_list: list):
        for url in tqdm(url_list, dynamic_ncols=True):
            if pd.isna(url):
                continue

            try:
                save_path = os.path.join(self.base_save_path, url)
                with requests.get(self.base_url + url, timeout=30, stream=True) as r:
                    r.raise_for_status()
                    open(save_path, "wb").write(r.content)
            except Exception as e:
                with open("error.log", "a") as f:
                    e_rep = str(e).replace("\n", " ")
                    log(f"[CRAWL ERROR] {url} {e_rep}")


class Preprocessor:
    def __init__(self, raw_data_csv_path: str):
        self.raw_df = pd.read_csv(raw_data_csv_path)
        self.processed_df = self.raw_df
        self.processed_data_list = None

    def clean_na(self):
        self.processed_df = self.processed_df.dropna(axis=1, how="all").where(
            pd.notna(self.processed_df), None
        )
        return self.processed_df

    def get_profile_image_list(self):
        profile_dict = self.processed_df[
            [col for col in self.processed_df if "pic" in col]
        ].T.to_dict()
        profile_image_list = [
            value
            for cur in profile_dict.values()
            for value in cur.values()
            if pd.notna(value)
        ]
        return profile_image_list

    def check_data_list_type(self, idx=0):
        for key, val in self.processed_data_list[idx].items():
            print(key, type(val))
            if isinstance(val, dict):
                for key, inner_val in val.items():
                    print("---", key, type(inner_val))

    def process_to_data_list(self):
        self.processed_data_list = []
        for idx, (_, row) in enumerate(self.processed_df.iterrows()):
            data = {
                "id": idx,
                "payload": {
                    "name_kr": validate(str(row["name_kr"])),
                    "name_en": validate(str(row["name_en"])),
                    "birth": convert_timestamp_to_date(row["birth"]),  # 체크필요
                    "gender": "남" if row["gender"] == 1 else "여",
                    "height": int(row["height"]),
                    "weight": int(row["weight"]),
                    "bodysize": validate(str(row["bodysize"])),
                    "lastedu": validate(str(row["lastedu"])),
                    "company": validate(str(row["company"])),
                    "job": validate(str(row["job"])),
                    "specialty": validate(str(row["specialty"])),
                    "specialty_etc": validate(str(row["specialty_etc"])),
                    "keyword": validate(str(row["keyword"])),
                    "field1": validate(str(row["field1"])),
                    "field2": validate(str(row["field2"])),
                    "field3": validate(str(row["field3"])),
                    "field4": validate(str(row["field4"])),
                    "last_work": validate(str(row["last_work"])),
                    "last_adver": validate(str(row["last_adver"])),
                    "sns_insta": validate(str(row["sns_insta"])),
                    "youtube": validate(str(row["youtube"])),
                    "videoPath": validate(str(row["videoPath"])),
                    "qr_youtube": validate(str(row["qr_youtube"])),
                    "comments": validate(str(row["comments"])),
                    "images": [
                        row[cur]
                        for cur in row.keys()
                        if "_pic" in cur
                        if row[cur] is not None
                    ],
                },
                "vector": {"face": None, "prompt": None},
            }
            meta = data["payload"]
            prompt_source = {
                "gender": meta["gender"],
                "height": meta["height"],
                "weight": meta["weight"],
                "bodysize": meta["bodysize"],
                "lastedu": meta["lastedu"],
                "company": meta["company"],
                "job": meta["job"],
                "specialty": validate(
                    " ".join(
                        str(meta[key])
                        for key in [
                            "specialty",
                            "specialty_etc",
                        ]
                        if meta.get(key)
                    )
                ),
                "tags": meta["keyword"],
                "field": validate(
                    " ".join(
                        str(meta[key])
                        for key in ["field1", "field2", "field3", "field4"]
                        if meta.get(key)
                    )
                ),
                "experience": validate(
                    " ".join(
                        str(meta[key])
                        for key in ["last_work", "last_adver"]
                        if meta.get(key)
                    )
                ),
                "comments": meta["comments"],
            }
            data["payload"]["prompt_source"] = prompt_source
            data["payload"]["image_url"] = (
                f"{CRAWL_BASE_URL}{data['payload']['images'][0]}"
                if data["payload"]["images"]
                else None
            )
            self.processed_data_list.append(data)
        return self.processed_data_list


if __name__ == "__main__":
    # crawler = Crawler(CRAWL_BASE_URL, CRAWL_DATA_PATH)
    preprocessor = Preprocessor("datas/member_202503311252.csv")
    preprocessor.clean_na()

    profile_image_list = preprocessor.get_profile_image_list()
    # crawler.run(profile_image_list)

    processed_data_list = preprocessor.process_to_data_list()

    face_model = FaceEmbeddingModel()
    db_client = VectorDBClient(VECTOR_DB_URL, VECTOR_DB_COLLECTION)
    import numpy as np
    for idx, data in tqdm(enumerate(processed_data_list), total=len(processed_data_list)):
        profile_images = [
            os.path.join(CRAWL_DATA_PATH, cur)
            for cur in data.get("payload").get("images")
        ]

        try:
            face_embeddings = face_model.run_model(
                profile_images, only_return_face=True
            )
        except Exception as e:
            log(f"[MODEL ERROR] {data['id']} / {str(e)}")
        
        prompt_embeddings = np.random.rand(768).tolist()

        db_client.insert(
            point_id=idx,
            vectors={"face":face_embeddings, "prompt":prompt_embeddings},
            payload=data["payload"],
        )
        data["vector"]["face"] = face_embeddings
        print(f"Success {data['id']}")
