import time
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .analyzer import analyze_driver_state
from .auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_driver,
    require_supervisor,
    user_public_from_record,
    verify_password,
)
from .config import config
from .face_metrics import face_metrics_extractor
from .gaze_inference import gaze_inference_service
from .schemas import (
    AdminDbOverviewResponse,
    AdminDbTableItem,
    AdminUserCreateRequest,
    AdminUserUpdateRequest,
    AdminUsersResponse,
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AuthResponse,
    DetectObjectsRequest,
    DetectObjectsResponse,
    DriverViolationItem,
    DriverViolationsResponse,
    DriverTelemetryRequest,
    DriverTelemetryResponse,
    FrameMetrics,
    GazeInferRequest,
    GazeInferResponse,
    LoginRequest,
    RegisterRequest,
    SupervisorDriverItem,
    SupervisorOverviewResponse,
    UpdateProfileRequest,
    UserPublic,
)
from .store import UserRecord, store
from .yolo_detector import detector

app = FastAPI(title=config.app_name, version=config.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def seed_default_users() -> None:
    store.init_db()

    defaults = [
        ("Administrator", "admin", "SUPERVISOR", "admin"),
        ("Supervisor Demo", "supervisor@dms.local", "SUPERVISOR", "123456"),
        ("Driver Demo", "driver@dms.local", "DRIVER", "123456"),
    ]

    for full_name, email, role, password in defaults:
        store.ensure_user(
            full_name=full_name,
            email=email,
            role=role,  # type: ignore[arg-type]
            password_hash=hash_password(password),
        )


@app.post(f"{config.api_prefix}/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    try:
        user = store.create_user(
            full_name=req.full_name,
            email=req.email,
            role="DRIVER",
            password_hash=hash_password(req.password),
            phone_number=req.phone_number,
            license_plate=req.license_plate,
        )
    except ValueError as ex:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(ex)) from ex

    token = create_access_token(subject=user.email, role=user.role)
    return AuthResponse(access_token=token, user=user_public_from_record(user))


@app.post(f"{config.api_prefix}/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    from fastapi import HTTPException

    user = store.get_user_by_email(req.email)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Sai email hoặc mật khẩu")

    token = create_access_token(subject=user.email, role=user.role)
    return AuthResponse(access_token=token, user=user_public_from_record(user))


@app.get(f"{config.api_prefix}/auth/me", response_model=UserPublic)
def me(user: Annotated[UserRecord, Depends(get_current_user)]) -> UserPublic:
    return user_public_from_record(user)


@app.put(f"{config.api_prefix}/auth/profile", response_model=AuthResponse)
def update_profile(
    req: UpdateProfileRequest,
    user: Annotated[UserRecord, Depends(require_driver)],
) -> AuthResponse:
    from fastapi import HTTPException

    try:
        updated = store.update_user_profile(
            user_id=user.id,
            full_name=req.full_name,
            email=req.email,
        )
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    token = create_access_token(subject=updated.email, role=updated.role)
    return AuthResponse(access_token=token, user=user_public_from_record(updated))


@app.post(f"{config.api_prefix}/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return analyze_driver_state(driver_id=req.driver_id, m=req.metrics)


@app.post(f"{config.api_prefix}/analyze-frame", response_model=AnalyzeFrameResponse)
def analyze_frame(req: AnalyzeFrameRequest) -> AnalyzeFrameResponse:
    face = face_metrics_extractor.extract(req.image_base64)
    detected = detector.detect(image_base64=req.image_base64, conf_threshold=req.conf_threshold)

    no_face_duration = 0.0 if bool(face.get("has_face")) else 1.8

    frame_metrics = FrameMetrics(
        ear=float(face.get("ear", 0.3)),
        mar=float(face.get("mar", 0.2)),
        closed_eye_duration=0.0,
        blink_per_min=0.0,
        head_pitch=float(face.get("head_pitch", 0.0)),
        head_yaw=float(face.get("head_yaw", 0.0)),
        gaze_zone=str(face.get("gaze_zone", "FRONT_WINDSHIELD")),
        attention_group=str(face.get("attention_group", "PRIMARY")),
        phone_detected=detected.flags.phone_detected,
        cigarette_detected=detected.flags.cigarette_detected,
        earphone_detected=detected.flags.earphone_detected,
        cup_detected=detected.flags.cup_detected,
        bottle_detected=detected.flags.bottle_detected,
        seatbelt_detected=detected.flags.seatbelt_detected,
        no_seatbelt_detected=detected.flags.no_seatbelt_detected,
        no_face_duration=no_face_duration,
        speed_kmh=req.speed_kmh,
    )

    analysis = analyze_driver_state(driver_id=req.driver_id, m=frame_metrics)
    return AnalyzeFrameResponse(
        analysis=analysis,
        metrics=frame_metrics,
        detections=detected.detections,
        flags=detected.flags,
    )


@app.post(f"{config.api_prefix}/gaze-infer", response_model=GazeInferResponse)
def gaze_infer(req: GazeInferRequest) -> GazeInferResponse:
    out = gaze_inference_service.infer(req.image_base64)
    return GazeInferResponse(
        gaze_zone=str(out.get("gaze_zone", "FRONT_WINDSHIELD")),
        attention_group=str(out.get("attention_group", "PRIMARY")),
        head_pitch=float(out.get("head_pitch", 0.0)),
        head_yaw=float(out.get("head_yaw", 0.0)),
        confidence=float(out.get("confidence", 0.0)),
        source=str(out.get("source", "fallback")),
        model_available=bool(out.get("model_available", False)),
    )


@app.post(f"{config.api_prefix}/detect-objects", response_model=DetectObjectsResponse)
def detect_objects(req: DetectObjectsRequest) -> DetectObjectsResponse:
    return detector.detect(image_base64=req.image_base64, conf_threshold=req.conf_threshold)



@app.post(f"{config.api_prefix}/driver/telemetry", response_model=DriverTelemetryResponse)
def driver_telemetry(
    req: DriverTelemetryRequest,
    user: Annotated[UserRecord, Depends(require_driver)],
) -> DriverTelemetryResponse:
    store.update_driver_runtime(
        driver_user=user,
        risk_score=req.risk_score,
        risk_level=req.risk_level,
        speed_kmh=req.speed_kmh,
        latitude=req.latitude,
        longitude=req.longitude,
        frame_base64=req.frame_base64,
    )

    store.create_driver_violation_if_needed(
        driver_user=user,
        risk_score=req.risk_score,
        risk_level=req.risk_level,
        threshold=config.critical_violation_threshold,
        cooldown_sec=config.critical_violation_cooldown_sec,
        latitude=req.latitude,
        longitude=req.longitude,
        location_source=req.location_source,
        frame_base64=req.frame_base64,
        event_codes=req.event_codes,
        event_messages=req.event_messages,
    )

    return DriverTelemetryResponse(ok=True)


@app.get(f"{config.api_prefix}/driver/violations", response_model=DriverViolationsResponse)
def driver_violations(
    user: Annotated[UserRecord, Depends(require_driver)],
) -> DriverViolationsResponse:
    rows = store.list_driver_violations(driver_id=user.id, limit=80)
    return DriverViolationsResponse(
        items=[
            DriverViolationItem(
                id=r.id,
                driver_id=r.driver_id,
                risk_score=r.risk_score,
                risk_level=r.risk_level,
                latitude=r.latitude,
                longitude=r.longitude,
                location_source=r.location_source,  # type: ignore[arg-type]
                event_codes=r.event_codes or [],
                event_messages=r.event_messages or [],
                frame_base64=r.frame_base64,
                created_at_unix_ms=r.created_at_unix_ms,
            )
            for r in rows
        ]
    )


@app.get(f"{config.api_prefix}/supervisor/overview", response_model=SupervisorOverviewResponse)
def supervisor_overview(
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> SupervisorOverviewResponse:
    _ = user
    now_ms = time.time() * 1000
    active_threshold_ms = 15_000

    items: list[SupervisorDriverItem] = []
    for d in store.list_driver_runtime():
        is_active = (now_ms - d.last_seen_unix_ms) <= active_threshold_ms
        items.append(
            SupervisorDriverItem(
                driver_id=d.driver_id,
                driver_name=d.driver_name,
                email=d.email,
                phone_number=d.phone_number,
                license_plate=d.license_plate,
                risk_score=d.risk_score,
                risk_level=d.risk_level,
                speed_kmh=d.speed_kmh,
                latitude=d.latitude,
                longitude=d.longitude,
                is_active=is_active,
                last_seen_unix_ms=d.last_seen_unix_ms,
                latest_frame_base64=d.latest_frame_base64,
            )
        )

    active_count = sum(1 for x in items if x.is_active)
    return SupervisorOverviewResponse(
        active_driver_count=active_count,
        total_driver_count=len(items),
        drivers=items,
    )


@app.get(f"{config.api_prefix}/admin/db-overview", response_model=AdminDbOverviewResponse)
def admin_db_overview(
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> AdminDbOverviewResponse:
    _ = user
    masked_url, tables = store.get_db_overview()
    return AdminDbOverviewResponse(
        database_url_masked=masked_url,
        tables=[AdminDbTableItem(table_name=name, row_count=count) for name, count in tables],
    )


@app.get(f"{config.api_prefix}/admin/users", response_model=AdminUsersResponse)
def admin_list_users(
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> AdminUsersResponse:
    _ = user
    return AdminUsersResponse(items=[user_public_from_record(u) for u in store.list_users()])


@app.post(f"{config.api_prefix}/admin/users", response_model=UserPublic)
def admin_create_user(
    req: AdminUserCreateRequest,
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> UserPublic:
    _ = user
    from fastapi import HTTPException

    try:
        created = store.create_user(
            full_name=req.full_name,
            email=req.email,
            role=req.role,
            password_hash=hash_password(req.password),
            phone_number=req.phone_number,
            license_plate=req.license_plate,
        )
        return user_public_from_record(created)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex


@app.put(f"{config.api_prefix}/admin/users/{{user_id}}", response_model=UserPublic)
def admin_update_user(
    user_id: str,
    req: AdminUserUpdateRequest,
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> UserPublic:
    from fastapi import HTTPException

    if user_id == user.id and req.role != "SUPERVISOR":
        raise HTTPException(status_code=400, detail="Không thể tự hạ quyền tài khoản hiện tại")

    try:
        updated = store.update_user_admin(
            user_id=user_id,
            full_name=req.full_name,
            email=req.email,
            role=req.role,
            phone_number=req.phone_number,
            license_plate=req.license_plate,
        )
        return user_public_from_record(updated)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex


@app.delete(f"{config.api_prefix}/admin/users/{{user_id}}", response_model=DriverTelemetryResponse)
def admin_delete_user(
    user_id: str,
    user: Annotated[UserRecord, Depends(require_supervisor)],
) -> DriverTelemetryResponse:
    from fastapi import HTTPException

    if user_id == user.id:
        raise HTTPException(status_code=400, detail="Không thể tự xóa tài khoản hiện tại")

    ok = store.delete_user_admin(user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại")
    return DriverTelemetryResponse(ok=True)
