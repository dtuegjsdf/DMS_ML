import os
from pathlib import Path

import numpy as np


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SeatbeltClassifier:
    def __init__(
        self,
        model_path: str,
        input_size: int = 224,
        threshold: float = 0.5,
        class_labels: list[str] | None = None,
        binary_output_is_seatbelt: bool = True,
        scale_01: bool = True,
    ) -> None:
        self._model = None
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.class_labels = class_labels or ["no seat belt", "seat belt"]
        self.binary_output_is_seatbelt = binary_output_is_seatbelt
        self.scale_01 = scale_01

        self._positive_aliases = {
            "seat belt",
            "seatbelt",
            "with seat belt",
            "buckled",
            "driver with seat belt",
        }
        self._negative_aliases = {
            "no seat belt",
            "without seat belt",
            "unbuckled",
            "no-seatbelt",
            "driver without seat belt",
        }

        self._load_model()

    @classmethod
    def from_env(cls) -> "SeatbeltClassifier | None":
        model_path = os.getenv("DMS_SEATBELT_KERAS_MODEL", "my_model.h5").strip()
        if not model_path:
            return None

        path_obj = Path(model_path)
        if not path_obj.is_absolute():
            path_obj = Path(__file__).resolve().parent.parent / path_obj
        path_obj = path_obj.expanduser().resolve()
        if not path_obj.exists():
            return None

        labels_env = os.getenv("DMS_SEATBELT_CLASS_LABELS", "")
        labels = [x.strip().lower() for x in labels_env.split(",") if x.strip()] if labels_env else None

        input_size = int(os.getenv("DMS_SEATBELT_INPUT_SIZE", "224"))
        threshold = float(os.getenv("DMS_SEATBELT_CLASSIFIER_THRESHOLD", "0.5"))
        binary_output_is_seatbelt = _to_bool(
            os.getenv("DMS_SEATBELT_BINARY_OUTPUT_IS_SEATBELT", "true"),
            default=True,
        )
        scale_01 = _to_bool(os.getenv("DMS_SEATBELT_SCALE_01", "true"), default=True)

        return cls(
            model_path=str(path_obj),
            input_size=input_size,
            threshold=threshold,
            class_labels=labels,
            binary_output_is_seatbelt=binary_output_is_seatbelt,
            scale_01=scale_01,
        )

    def _load_model(self) -> None:
        try:
            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)
        except Exception:
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        resized = Image.fromarray(image).resize((self.input_size, self.input_size))
        tensor = np.asarray(resized, dtype=np.float32)
        if self.scale_01:
            tensor = tensor / 255.0
        return np.expand_dims(tensor, axis=0)

    def predict(self, image: np.ndarray) -> tuple[str | None, float]:
        if self._model is None:
            return None, 0.0

        try:
            x = self._preprocess(image)
            y = self._model.predict(x, verbose=0)
        except Exception:
            return None, 0.0

        arr = np.asarray(y)
        if arr.ndim == 0:
            prob = float(arr)
            seatbelt_prob = prob if self.binary_output_is_seatbelt else (1.0 - prob)
            if seatbelt_prob >= self.threshold:
                return "seat belt", seatbelt_prob
            return "no seat belt", 1.0 - seatbelt_prob

        flat = arr.reshape(-1)
        if flat.size == 1:
            prob = float(flat[0])
            seatbelt_prob = prob if self.binary_output_is_seatbelt else (1.0 - prob)
            if seatbelt_prob >= self.threshold:
                return "seat belt", seatbelt_prob
            return "no seat belt", 1.0 - seatbelt_prob

        idx = int(np.argmax(flat))
        confidence = float(flat[idx])
        if idx < len(self.class_labels):
            label = self.class_labels[idx]
        else:
            label = f"class_{idx}"
        return label.strip().lower(), confidence

    def to_flag_name(self, label: str | None) -> str | None:
        if not label:
            return None
        normalized = label.strip().lower().replace("_", " ")
        if normalized in self._positive_aliases:
            return "seatbelt_detected"
        if normalized in self._negative_aliases:
            return "no_seatbelt_detected"
        return None
