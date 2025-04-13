import io
from typing import Tuple, Union, List, Optional
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace

class FaceEmbeddingModel:
    def __init__(self):
        self.deepface = DeepFace
        self.model_name = "Facenet512"
        self.detector_backend = "retinaface"
        self.confidence_threshold = 0.5

    def _convert_to_numpy(self, image_input: Union[str, io.BytesIO]) -> np.ndarray:
        """이미지 입력을 BGR 형식의 numpy 배열로 변환합니다."""
        try:
            if isinstance(image_input, str):
                img_bgr = cv2.imread(image_input)
                if img_bgr is None:
                    raise ValueError(f"Could not read image at {image_input}")
                return img_bgr
            elif isinstance(image_input, io.BytesIO):
                img = Image.open(image_input)
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        except Exception as e:
            raise ValueError(f"Failed to convert image to numpy array: {str(e)}")

    def _process_image_list(self, image_list: List[Union[str, io.BytesIO]]) -> List[np.ndarray]:
        """이미지 리스트를 BGR 형식의 numpy 배열 리스트로 변환합니다."""
        return [self._convert_to_numpy(img) for img in image_list]

    def _get_embedding(self, image: np.ndarray) -> List[dict]:
        """단일 이미지에 대한 임베딩을 추출합니다."""
        return self.deepface.represent(
            img_path=image,
            enforce_detection=False,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
        )

    def _process_embedding_results(self, embedding_objs: List[dict], only_return_face: bool) -> List[Tuple[float, list]]:
        """임베딩 결과를 처리하여 (confidence, embedding) 튜플 리스트를 반환합니다."""
        results = []
        for embedding_obj in embedding_objs:
            if isinstance(embedding_obj, list):
                results.extend((cur.get("face_confidence"), cur.get("embedding")) for cur in embedding_obj)
            else:
                results.append((embedding_obj.get("face_confidence"), embedding_obj.get("embedding")))

        if only_return_face:
            return [embed for conf, embed in results if conf > self.confidence_threshold]
        return results

    def run_model(
            self,
            image_input: Union[str, io.BytesIO, List[Union[str, io.BytesIO]]],
            only_return_face: bool = False
        ) -> List[Tuple[float, list]]:
        try:
            # 입력 이미지 처리
            processed_images = (
                self._process_image_list(image_input)
                if isinstance(image_input, list)
                else [self._convert_to_numpy(image_input)]
            )

            # 각 이미지에 대한 임베딩 추출
            embedding_objs = [self._get_embedding(img) for img in processed_images]

            # 결과 처리 및 반환
            return self._process_embedding_results(embedding_objs, only_return_face)
            
        except Exception as e:
            raise RuntimeError(f"Failed to run face embedding model: {str(e)}")

class PromptEmbeddingModel:
    pass
