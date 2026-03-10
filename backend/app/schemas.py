from typing import Literal
from pydantic import BaseModel, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
UserRole = Literal["SUPERVISOR", "DRIVER"]


class FrameMetrics(BaseModel):
    ear: float = Field(..., ge=0.0, le=1.0, description="Eye Aspect Ratio")
    mar: float = Field(..., ge=0.0, le=1.0, description="Mouth Aspect Ratio")
    closed_eye_duration: float = Field(
        0.0, ge=0.0, description="Số giây mắt nhắm liên tục"
    )
    blink_per_min: float = Field(0.0, ge=0.0, description="Số lần nháy mắt/phút")
    head_pitch: float = Field(0.0, description="Góc cúi/ngửa đầu (độ)")
    head_yaw: float = Field(0.0, description="Góc quay trái/phải của đầu (độ)")
    gaze_zone: str = Field("FRONT_WINDSHIELD", description="Vùng nhìn hiện tại của tài xế")
    attention_group: str = Field("PRIMARY", description="Nhóm vùng nhìn: PRIMARY hoặc SECONDARY")
    phone_detected: bool = False
    cigarette_detected: bool = False
    earphone_detected: bool = False
    cup_detected: bool = False
    bottle_detected: bool = False
    seatbelt_detected: bool = False
    no_seatbelt_detected: bool = False
    no_face_duration: float = Field(0.0, ge=0.0, description="Số giây mất mặt khỏi camera")
    speed_kmh: float = Field(0.0, ge=0.0, description="Tốc độ xe (nếu có)")


class AnalyzeRequest(BaseModel):
    driver_id: str = Field(..., min_length=1)
    metrics: FrameMetrics


class DriverEvent(BaseModel):
    code: str
    message: str
    severity: RiskLevel


class AnalyzeResponse(BaseModel):
    driver_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    events: list[DriverEvent]


class DetectObjectsRequest(BaseModel):
    image_base64: str = Field(..., min_length=10)
    conf_threshold: float = Field(0.22, ge=0.05, le=0.95)


class ObjectDetection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: list[float]


class ObjectFlags(BaseModel):
    phone_detected: bool = False
    cigarette_detected: bool = False
    earphone_detected: bool = False
    cup_detected: bool = False
    bottle_detected: bool = False
    seatbelt_detected: bool = False
    no_seatbelt_detected: bool = False


class DetectObjectsResponse(BaseModel):
    detections: list[ObjectDetection]
    flags: ObjectFlags


class AnalyzeFrameRequest(BaseModel):
    driver_id: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=10)
    speed_kmh: float = Field(0.0, ge=0.0)
    conf_threshold: float = Field(0.22, ge=0.05, le=0.95)


class AnalyzeFrameResponse(BaseModel):
    analysis: AnalyzeResponse
    metrics: FrameMetrics
    detections: list[ObjectDetection]
    flags: ObjectFlags


class GazeInferRequest(BaseModel):
    image_base64: str = Field(..., min_length=10)


class GazeInferResponse(BaseModel):
    gaze_zone: str = Field("FRONT_WINDSHIELD")
    attention_group: str = Field("PRIMARY")
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str = Field("fallback")
    model_available: bool = False


class UserPublic(BaseModel):
    id: str
    full_name: str
    email: str
    role: UserRole
    phone_number: str | None = None
    license_plate: str | None = None


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    phone_number: str = Field(..., min_length=8, max_length=20)
    license_plate: str = Field(..., min_length=5, max_length=20)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class UpdateProfileRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=255)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class DriverTelemetryRequest(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    speed_kmh: float = Field(0.0, ge=0.0)
    latitude: float | None = Field(None, ge=-90.0, le=90.0)
    longitude: float | None = Field(None, ge=-180.0, le=180.0)
    location_source: Literal["gps", "manual"] | None = None
    frame_base64: str | None = None
    event_codes: list[str] = Field(default_factory=list)
    event_messages: list[str] = Field(default_factory=list)


class DriverTelemetryResponse(BaseModel):
    ok: bool


class DriverViolationItem(BaseModel):
    id: int
    driver_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    latitude: float | None = None
    longitude: float | None = None
    location_source: Literal["gps", "manual"] | None = None
    event_codes: list[str] = Field(default_factory=list)
    event_messages: list[str] = Field(default_factory=list)
    frame_base64: str | None = None
    created_at_unix_ms: int


class DriverViolationsResponse(BaseModel):
    items: list[DriverViolationItem]


class SupervisorDriverItem(BaseModel):
    driver_id: str
    driver_name: str
    email: str
    phone_number: str | None = None
    license_plate: str | None = None
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    speed_kmh: float
    latitude: float | None = None
    longitude: float | None = None
    is_active: bool
    last_seen_unix_ms: int
    latest_frame_base64: str | None = None


class SupervisorOverviewResponse(BaseModel):
    active_driver_count: int
    total_driver_count: int
    drivers: list[SupervisorDriverItem]


class AdminDbTableItem(BaseModel):
    table_name: str
    row_count: int


class AdminDbOverviewResponse(BaseModel):
    database_url_masked: str
    tables: list[AdminDbTableItem]


class AdminUserCreateRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    role: UserRole
    phone_number: str | None = Field(None, max_length=20)
    license_plate: str | None = Field(None, max_length=20)


class AdminUserUpdateRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=3, max_length=255)
    role: UserRole
    phone_number: str | None = Field(None, max_length=20)
    license_plate: str | None = Field(None, max_length=20)


class AdminUsersResponse(BaseModel):
    items: list[UserPublic]
