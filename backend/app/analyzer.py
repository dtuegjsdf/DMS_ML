from .schemas import AnalyzeResponse, DriverEvent, FrameMetrics


def _risk_level_from_score(score: int) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def analyze_driver_state(driver_id: str, m: FrameMetrics) -> AnalyzeResponse:
    score = 0
    events: list[DriverEvent] = []

    zone = m.gaze_zone.upper()
    yaw = m.head_yaw
    pitch = m.head_pitch
    sleep_slump_detected = m.ear <= 0.2 and pitch >= 28

    # Priority rule: eyes closed + head slumped down => driver likely dozing off.
    if sleep_slump_detected:
        score += 65
        events.append(
            DriverEvent(
                code="DRIVER_ASLEEP_SLUMP",
                message="Phat hien mat nham va dau guc xuong: tai xe co dau hieu ngu guc.",
                severity="CRITICAL",
            )
        )

    # 0) Drowsiness: prolonged eye closure, abnormal blink, yawning.
    if m.closed_eye_duration >= 2.0 and not sleep_slump_detected:
        score += 45
        events.append(
            DriverEvent(
                code="DROWSY_EYES_CLOSED",
                message="Phat hien mat nham lau, co dau hieu buon ngu.",
                severity="CRITICAL",
            )
        )
    elif m.ear <= 0.2 and not sleep_slump_detected:
        score += 18
        events.append(
            DriverEvent(
                code="LOW_EAR_DROWSY",
                message="Mat mo thap bat thuong, canh bao met moi/ngu gat.",
                severity="MEDIUM",
            )
        )

    if m.blink_per_min >= 35:
        score += 12
        events.append(
            DriverEvent(
                code="ABNORMAL_BLINK_RATE",
                message="Tan suat nhay mat cao bat thuong.",
                severity="MEDIUM",
            )
        )

    if m.mar >= 0.62:
        score += 24
        events.append(
            DriverEvent(
                code="YAWNING",
                message="Phat hien ngap, co dau hieu buon ngu/met moi.",
                severity="HIGH",
            )
        )

    # 1) Phone call (left/right): phone detected, but not in texting posture.
    if m.phone_detected and pitch < 18:
        if yaw <= 0:
            score += 55
            events.append(
                DriverEvent(
                    code="PHONE_CALL_LEFT",
                    message="Phat hien noi chuyen dien thoai ben trai khi lai xe.",
                    severity="CRITICAL",
                )
            )
        else:
            score += 55
            events.append(
                DriverEvent(
                    code="PHONE_CALL_RIGHT",
                    message="Phat hien noi chuyen dien thoai ben phai khi lai xe.",
                    severity="CRITICAL",
                )
            )

    # 2) Texting (left/right): phone detected with lowered head posture.
    if m.phone_detected and pitch >= 18:
        if yaw <= 0:
            score += 70
            events.append(
                DriverEvent(
                    code="TEXTING_LEFT",
                    message="Phat hien nhan tin dien thoai ben trai khi lai xe.",
                    severity="CRITICAL",
                )
            )
        else:
            score += 70
            events.append(
                DriverEvent(
                    code="TEXTING_RIGHT",
                    message="Phat hien nhan tin dien thoai ben phai khi lai xe.",
                    severity="CRITICAL",
                )
            )

    # 3) Talking to passenger: turning left/right/behind while not using phone.
    talking_passenger = (
        (not m.phone_detected)
        and (
            abs(yaw) >= 18
            or zone in {"LEFT_WINDOW", "RIGHT_WINDOW"}
            or zone == "BEHIND"
        )
    )
    if talking_passenger:
        score += 38
        events.append(
            DriverEvent(
                code="TALKING_TO_PASSENGER",
                message="Phat hien xoay dau noi chuyen voi hanh khach (trai/phai/phia sau).",
                severity="HIGH",
            )
        )

    # 4) Head nodding downward when attention is off primary road scan.
    if m.head_pitch >= 30 and m.attention_group.upper() == "SECONDARY" and not sleep_slump_detected:
        score += 16
        events.append(
            DriverEvent(
                code="HEAD_NODDING",
                message="Phat hien cui dau khi tam nhin roi khoi huong chinh.",
                severity="HIGH",
            )
        )

    # 5) Other risky actions during driving.
    if m.cigarette_detected:
        score += 26
        events.append(
            DriverEvent(
                code="SMOKING_DETECTED",
                message="Phat hien hut thuoc khi lai xe.",
                severity="HIGH",
            )
        )

    if m.earphone_detected:
        score += 14
        events.append(
            DriverEvent(
                code="EARPHONE_USAGE",
                message="Phat hien deo tai nghe khi lai xe.",
                severity="MEDIUM",
            )
        )

    if (m.cup_detected or m.bottle_detected) and m.speed_kmh >= 10:
        held = "coc/ly" if m.cup_detected else "chai nuoc"
        score += 12
        events.append(
            DriverEvent(
                code="HANDHELD_OBJECT",
                message=f"Phat hien cam {held} khi xe dang di chuyen.",
                severity="MEDIUM",
            )
        )

    if m.no_seatbelt_detected and m.speed_kmh >= 10:
        score += 30
        events.append(
            DriverEvent(
                code="NO_SEATBELT",
                message="Phat hien khong deo day an toan khi xe dang di chuyen.",
                severity="CRITICAL",
            )
        )
    elif m.no_seatbelt_detected:
        score += 20
        events.append(
            DriverEvent(
                code="NO_SEATBELT",
                message="Phat hien khong deo day an toan.",
                severity="HIGH",
            )
        )

    if m.no_face_duration >= 1.5:
        score += 18
        events.append(
            DriverEvent(
                code="NO_FACE",
                message="Mat mat khoi camera trong thoi gian dai.",
                severity="HIGH",
            )
        )

    # Điều chỉnh theo tốc độ xe (đi nhanh thì tăng rủi ro)
    if m.speed_kmh >= 60 and score > 0:
        score = int(score * 1.15)

    score = min(score, 100)
    risk_level = _risk_level_from_score(score)

    return AnalyzeResponse(
        driver_id=driver_id,
        risk_score=score,
        risk_level=risk_level,  # type: ignore[arg-type]
        events=events,
    )
