import base64
import threading
from io import BytesIO
import logging

import mediapipe as mp
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class FaceMetricsExtractor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._face_mesh = None
        try:
            # MediaPipe FaceMesh legacy API (mp.solutions) may be unavailable on
            # unsupported Python versions (e.g. 3.14) depending on wheel/runtime.
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as ex:
            self._face_mesh = None
            logger.warning(
                "MediaPipe FaceMesh unavailable, fallback to no-face metrics. "
                "Install supported Python (3.10-3.12) for full face metrics. Error: %s",
                ex,
            )

    @staticmethod
    def _decode_image(image_base64: str) -> np.ndarray:
        payload = image_base64
        if "," in payload:
            payload = payload.split(",", 1)[1]

        binary = base64.b64decode(payload)
        img = Image.open(BytesIO(binary)).convert("RGB")
        return np.array(img)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx * dx + dy * dy) ** 0.5)

    def _calc_ear(self, pts: list[tuple[float, float]], idx: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
        up = self._distance(p2, p6) + self._distance(p3, p5)
        down = 2 * self._distance(p1, p4)
        if down <= 1e-6:
            return 0.0
        return up / down

    def _calc_mar(self, pts: list[tuple[float, float]]) -> float:
        upper = pts[13]
        lower = pts[14]
        left = pts[78]
        right = pts[308]
        vertical = self._distance(upper, lower)
        horizontal = self._distance(left, right)
        if horizontal <= 1e-6:
            return 0.0
        return vertical / horizontal

    @staticmethod
    def _calc_head_pitch(pts: list[tuple[float, float]]) -> float:
        nose = pts[1]
        chin = pts[152]
        left_eye = pts[33]
        right_eye = pts[263]

        interocular = max(1e-6, FaceMetricsExtractor._distance(left_eye, right_eye))
        chin_to_nose = (chin[1] - nose[1]) / interocular

        # Neutral frontal pose usually sits around this ratio. We only keep
        # the positive residual to represent downward head motion.
        neutral_ratio = 1.12
        pitch = max(0.0, (chin_to_nose - neutral_ratio) * 70.0)
        return min(45.0, pitch)

    @staticmethod
    def _estimate_head_yaw(pts: list[tuple[float, float]]) -> float:
        left = pts[234]
        right = pts[454]
        nose = pts[1]
        half_face = max(0.0001, abs(right[0] - left[0]) / 2)
        mid = (left[0] + right[0]) / 2
        normalized = (nose[0] - mid) / half_face
        # Expand yaw dynamic range so the "BEHIND" threshold (>45 deg)
        # can be reached by heuristic estimation.
        yaw = normalized * 60
        return max(-60.0, min(60.0, yaw))

    @staticmethod
    def _classify_gaze(head_yaw: float, head_pitch: float) -> tuple[str, str]:
        yaw_r = head_yaw

        if abs(yaw_r) > 45:
            return "BEHIND", "SECONDARY"

        if abs(yaw_r) >= 10:
            if yaw_r < 0:
                return "LEFT_WINDOW", "SECONDARY"
            return "RIGHT_WINDOW", "SECONDARY"

        return "FRONT_WINDSHIELD", "PRIMARY"

    def extract(self, image_base64: str) -> dict[str, float | str | bool]:
        if self._face_mesh is None:
            return {
                "ear": 0.3,
                "mar": 0.2,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "gaze_zone": "FRONT_WINDSHIELD",
                "attention_group": "PRIMARY",
                "has_face": False,
            }

        frame = self._decode_image(image_base64)

        with self._lock:
            result = self._face_mesh.process(frame)

        if not result.multi_face_landmarks:
            return {
                "ear": 0.3,
                "mar": 0.2,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "gaze_zone": "FRONT_WINDSHIELD",
                "attention_group": "PRIMARY",
                "has_face": False,
            }

        lms = result.multi_face_landmarks[0].landmark
        pts = [(float(p.x), float(p.y)) for p in lms]

        left_ear = self._calc_ear(pts, [33, 160, 158, 133, 153, 144])
        right_ear = self._calc_ear(pts, [362, 385, 387, 263, 373, 380])
        ear = max(0.0, min(1.0, (left_ear + right_ear) / 2))
        mar = max(0.0, min(1.0, self._calc_mar(pts)))
        head_pitch = self._calc_head_pitch(pts)
        head_yaw = self._estimate_head_yaw(pts)
        gaze_zone, attention_group = self._classify_gaze(head_yaw, head_pitch)

        return {
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "head_pitch": round(head_pitch, 2),
            "head_yaw": round(head_yaw, 2),
            "gaze_zone": gaze_zone,
            "attention_group": attention_group,
            "has_face": True,
        }


face_metrics_extractor = FaceMetricsExtractor()
