import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .seatbelt_classifier import SeatbeltClassifier
from .schemas import DetectObjectsResponse, ObjectDetection, ObjectFlags

GENERAL_FLAG_LABEL_ALIASES: dict[str, set[str]] = {
    "phone_detected": {"cell phone", "mobile phone", "phone", "smartphone"},
    "cigarette_detected": {"cigarette", "smoke", "vape", "e-cigarette"},
    "earphone_detected": {"earphone", "earphones", "earbud", "earbuds", "airpods", "headset", "headphones"},
    "cup_detected": {"cup"},
    "bottle_detected": {"bottle"},
}

SEATBELT_FLAG_LABEL_ALIASES: dict[str, set[str]] = {
    "seatbelt_detected": {
        "seat belt",
        "seatbelt",
        "belt",
        "with seat belt",
        "seat belt on",
        "buckled",
        "driver with seat belt",
    },
    "no_seatbelt_detected": {
        "no seat belt",
        "without seat belt",
        "unbuckled",
        "no-seatbelt",
        "seat belt off",
        "driver without seat belt",
    },
}


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


class YoloV8sDetector:
    def __init__(self, model_name: str = "yolov8x.pt", seatbelt_model_name: str | None = None) -> None:
        self.model = YOLO(model_name)
        self.seatbelt_model = self._load_optional_seatbelt_model(seatbelt_model_name)
        self.seatbelt_classifier = SeatbeltClassifier.from_env()
        self.seatbelt_classifier_min_conf = float(os.getenv("DMS_SEATBELT_CLASSIFIER_MIN_CONF", "0.75"))
        self.seatbelt_min_conf = float(os.getenv("DMS_SEATBELT_MIN_CONF", "0.55"))
        self.no_seatbelt_min_conf = float(os.getenv("DMS_NO_SEATBELT_MIN_CONF", "0.72"))
        self.seatbelt_tie_margin = float(os.getenv("DMS_SEATBELT_TIE_MARGIN", "0.08"))

    @staticmethod
    def _workspace_root() -> Path:
        return Path(__file__).resolve().parent.parent

    @classmethod
    def _default_seatbelt_model_path(cls) -> Path:
        return cls._workspace_root() / "runs" / "runs" / "detect" / "train5" / "weights" / "best.pt"

    @classmethod
    def _load_optional_seatbelt_model(cls, seatbelt_model_name: str | None) -> YOLO | None:
        candidate = seatbelt_model_name.strip() if seatbelt_model_name else ""
        if not candidate:
            default_path = cls._default_seatbelt_model_path()
            candidate = str(default_path) if default_path.exists() else ""

        if not candidate:
            return None

        path_like = Path(candidate)
        if path_like.suffix:
            path_like = path_like.expanduser().resolve()
            if not path_like.exists():
                return None
            candidate = str(path_like)

        try:
            return YOLO(candidate)
        except Exception:
            return None

    @staticmethod
    def _decode_image(image_base64: str) -> np.ndarray:
        payload = image_base64
        if "," in image_base64:
            payload = image_base64.split(",", 1)[1]

        binary = base64.b64decode(payload)
        img = Image.open(BytesIO(binary)).convert("RGB")
        return np.array(img)

    @staticmethod
    def _collect_detections(
        result: object,
        label_aliases: dict[str, set[str]],
        detections: list[ObjectDetection],
        flags: ObjectFlags,
    ) -> None:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None)
        if boxes is None or not isinstance(names, dict):
            return

        for box in boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, "unknown"))
            normalized = _normalize_label(label)

            matched_flag: str | None = None
            for flag_name, aliases in label_aliases.items():
                if normalized in aliases:
                    matched_flag = flag_name
                    break

            if matched_flag is None:
                continue

            confidence = float(box.conf.item())
            coords = box.xyxy[0].tolist()

            detections.append(
                ObjectDetection(
                    label=normalized,
                    confidence=round(confidence, 4),
                    bbox_xyxy=[float(round(v, 2)) for v in coords],
                )
            )
            setattr(flags, matched_flag, True)

    def _seatbelt_confidence_from_detections(self, detections: list[ObjectDetection]) -> tuple[float, float]:
        seatbelt_conf = 0.0
        no_seatbelt_conf = 0.0

        for detection in detections:
            label = _normalize_label(detection.label)
            if label in SEATBELT_FLAG_LABEL_ALIASES["seatbelt_detected"]:
                conf = float(detection.confidence)
                if conf >= self.seatbelt_min_conf:
                    seatbelt_conf = max(seatbelt_conf, conf)
            elif label in SEATBELT_FLAG_LABEL_ALIASES["no_seatbelt_detected"]:
                conf = float(detection.confidence)
                if conf >= self.no_seatbelt_min_conf:
                    no_seatbelt_conf = max(no_seatbelt_conf, conf)

        return seatbelt_conf, no_seatbelt_conf

    def detect(self, image_base64: str, conf_threshold: float = 0.35) -> DetectObjectsResponse:
        image = self._decode_image(image_base64)

        detections: list[ObjectDetection] = []
        flags = ObjectFlags()
        combined_aliases = {**GENERAL_FLAG_LABEL_ALIASES, **SEATBELT_FLAG_LABEL_ALIASES}

        base_results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        if base_results:
            self._collect_detections(
                result=base_results[0],
                label_aliases=combined_aliases,
                detections=detections,
                flags=flags,
            )

        if self.seatbelt_model is not None:
            seatbelt_results = self.seatbelt_model.predict(source=image, conf=conf_threshold, verbose=False)
            if seatbelt_results:
                self._collect_detections(
                    result=seatbelt_results[0],
                    label_aliases=SEATBELT_FLAG_LABEL_ALIASES,
                    detections=detections,
                    flags=flags,
                )

        if self.seatbelt_classifier is not None and self.seatbelt_classifier.is_ready:
            label, confidence = self.seatbelt_classifier.predict(image)
            mapped_flag = self.seatbelt_classifier.to_flag_name(label)
            if mapped_flag is not None and float(confidence) >= self.seatbelt_classifier_min_conf:
                h, w = image.shape[:2]
                canonical_label = "seat belt" if mapped_flag == "seatbelt_detected" else "no seat belt"
                detections.append(
                    ObjectDetection(
                        label=canonical_label,
                        confidence=round(float(confidence), 4),
                        bbox_xyxy=[0.0, 0.0, float(w), float(h)],
                    )
                )

        seatbelt_conf, no_seatbelt_conf = self._seatbelt_confidence_from_detections(detections)

        if seatbelt_conf == 0.0 and no_seatbelt_conf == 0.0:
            flags.seatbelt_detected = False
            flags.no_seatbelt_detected = False
        elif seatbelt_conf > 0.0 and no_seatbelt_conf == 0.0:
            flags.seatbelt_detected = True
            flags.no_seatbelt_detected = False
        elif no_seatbelt_conf > 0.0 and seatbelt_conf == 0.0:
            flags.no_seatbelt_detected = True
            flags.seatbelt_detected = False
        else:
            if seatbelt_conf + self.seatbelt_tie_margin >= no_seatbelt_conf:
                flags.seatbelt_detected = True
                flags.no_seatbelt_detected = False
            else:
                flags.no_seatbelt_detected = True
                flags.seatbelt_detected = False

        return DetectObjectsResponse(detections=detections, flags=flags)


MODEL_NAME = os.getenv("DMS_YOLO_MODEL", "yolov8x.pt")
SEATBELT_MODEL_NAME = os.getenv("DMS_SEATBELT_MODEL", "")
detector = YoloV8sDetector(MODEL_NAME, seatbelt_model_name=SEATBELT_MODEL_NAME or None)
