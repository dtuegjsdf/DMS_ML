import importlib
import logging
from typing import Any

from .config import config
from .face_metrics import face_metrics_extractor


logger = logging.getLogger(__name__)


class GazeInferenceService:
    def __init__(self) -> None:
        self._infer_fn: Any = None
        self._model_available = False

        if not config.l2cs_enabled:
            return

        if not config.l2cs_adapter_module:
            logger.warning("L2CS enabled but L2CS_ADAPTER_MODULE is empty. Using fallback gaze inference.")
            return

        try:
            module = importlib.import_module(config.l2cs_adapter_module)
            infer_fn = getattr(module, config.l2cs_adapter_function)
            if not callable(infer_fn):
                raise TypeError("Configured L2CS adapter is not callable")
            self._infer_fn = infer_fn
            self._model_available = True
        except Exception as ex:
            logger.warning("Failed to load L2CS adapter. Using fallback gaze inference. Error: %s", ex)
            self._infer_fn = None
            self._model_available = False

    @property
    def model_available(self) -> bool:
        return self._model_available

    def infer(self, image_base64: str) -> dict[str, float | str | bool]:
        # Optional L2CS adapter path. Adapter should return a dict with gaze_zone,
        # attention_group, head_yaw, head_pitch, and confidence keys.
        if self._infer_fn is not None:
            try:
                result = self._infer_fn(image_base64)
                if isinstance(result, dict) and "gaze_zone" in result and "attention_group" in result:
                    return {
                        "gaze_zone": str(result.get("gaze_zone", "FRONT_WINDSHIELD")),
                        "attention_group": str(result.get("attention_group", "PRIMARY")),
                        "head_pitch": float(result.get("head_pitch", 0.0)),
                        "head_yaw": float(result.get("head_yaw", 0.0)),
                        "confidence": float(result.get("confidence", 0.5)),
                        "source": "l2cs",
                        "model_available": True,
                    }
            except Exception as ex:
                logger.warning("L2CS adapter inference failed. Falling back to heuristic gaze. Error: %s", ex)

        fallback = face_metrics_extractor.extract(image_base64)
        return {
            "gaze_zone": str(fallback.get("gaze_zone", "FRONT_WINDSHIELD")),
            "attention_group": str(fallback.get("attention_group", "PRIMARY")),
            "head_pitch": float(fallback.get("head_pitch", 0.0)),
            "head_yaw": float(fallback.get("head_yaw", 0.0)),
            "confidence": 0.35,
            "source": "fallback",
            "model_available": self._model_available,
        }


gaze_inference_service = GazeInferenceService()
