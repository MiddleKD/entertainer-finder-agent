import io
from typing import Tuple, Union, List

class FaceEmbeddingModel:
    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace

    def run_model(
            self,
            image_bytes_or_path: Union[str, io.BytesIO, List[Union[str, io.BytesIO]]],
            only_return_face:bool = False
        ) -> Union[Tuple[float, list], List]:
        
        embedding_objs = self.DeepFace.represent(
            img_path = image_bytes_or_path,
            enforce_detection=False,
            model_name = "Facenet512",
            detector_backend = "retinaface",
        )
        
        results = []
        for embedding_obj in embedding_objs:
            if isinstance(embedding_obj, list):
                for cur in embedding_obj:
                    results.append((cur.get("face_confidence"), cur.get("embedding")))
            else:
                cur = embedding_obj
                results.append((cur.get("face_confidence"), cur.get("embedding")))

        if only_return_face == True:
            return [embed for conf, embed in results if conf > 0.5]
        return results

class PromptEmbeddingModel:
    pass
