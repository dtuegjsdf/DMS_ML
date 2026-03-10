from __future__ import annotations

import os
import time
import uuid
import json
from dataclasses import dataclass

from sqlalchemy import BigInteger, Boolean, Float, Integer, String, create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .schemas import RiskLevel, UserRole


@dataclass
class UserRecord:
    id: str
    full_name: str
    email: str
    role: UserRole
    password_hash: str
    phone_number: str | None = None
    license_plate: str | None = None


@dataclass
class DriverRuntime:
    driver_id: str
    driver_name: str
    email: str
    phone_number: str | None = None
    license_plate: str | None = None
    risk_score: int = 0
    risk_level: RiskLevel = "LOW"
    speed_kmh: float = 0.0
    latitude: float | None = None
    longitude: float | None = None
    last_seen_unix_ms: int = 0
    latest_frame_base64: str | None = None


@dataclass
class DriverViolation:
    id: int
    driver_id: str
    risk_score: int
    risk_level: RiskLevel
    latitude: float | None = None
    longitude: float | None = None
    location_source: str | None = None
    event_codes: list[str] | None = None
    event_messages: list[str] | None = None
    frame_base64: str | None = None
    created_at_unix_ms: int = 0


class Base(DeclarativeBase):
    pass


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(80), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    license_plate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    face_embedding: Mapped[str | None] = mapped_column(String, nullable=True)


class DriverRuntimeORM(Base):
    __tablename__ = "driver_runtime"

    driver_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    driver_name: Mapped[str] = mapped_column(String(80), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    license_plate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, default="LOW")
    speed_kmh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    latest_frame_base64: Mapped[str | None] = mapped_column(String, nullable=True)


class DriverViolationORM(Base):
    __tablename__ = "driver_violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    event_codes: Mapped[str | None] = mapped_column(String, nullable=True)
    event_messages: Mapped[str | None] = mapped_column(String, nullable=True)
    frame_base64: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class VehicleORM(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    license_plate: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    brand: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model: Mapped[str | None] = mapped_column(String(40), nullable=True)
    color: Mapped[str | None] = mapped_column(String(24), nullable=True)
    production_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class TripLogORM(Base):
    __tablename__ = "trip_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    vehicle_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    started_at_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    ended_at_unix_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    start_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance_km: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="COMPLETED")


class AlertHistoryORM(Base):
    __tablename__ = "alert_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    trip_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    event_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_message: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class SystemSettingORM(Base):
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_key: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    config_value: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at_unix_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class PostgresStore:
    def __init__(self) -> None:
        self.database_url = os.getenv(
            "DATABASE_URL",
            "sqlite:///./dms.db",
        )
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, pool_pre_ping=True, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)
        inspector = inspect(self.engine)
        users_columns = {c["name"] for c in inspector.get_columns("users")}
        runtime_columns = {c["name"] for c in inspector.get_columns("driver_runtime")}
        violations_columns = {c["name"] for c in inspector.get_columns("driver_violations")} if inspector.has_table("driver_violations") else set()
        dialect = self.engine.dialect.name

        with self.engine.begin() as conn:
            if dialect == "postgresql" and "last_seen_unix_ms" in runtime_columns:
                conn.execute(
                    text(
                        "ALTER TABLE driver_runtime "
                        "ALTER COLUMN last_seen_unix_ms TYPE BIGINT"
                    )
                )

            if "face_embedding" not in users_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN face_embedding TEXT"))
            if "phone_number" not in users_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)"))
            if "license_plate" not in users_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN license_plate VARCHAR(20)"))

            if "phone_number" not in runtime_columns:
                conn.execute(text("ALTER TABLE driver_runtime ADD COLUMN phone_number VARCHAR(20)"))
            if "license_plate" not in runtime_columns:
                conn.execute(text("ALTER TABLE driver_runtime ADD COLUMN license_plate VARCHAR(20)"))
            if "latitude" not in runtime_columns:
                conn.execute(text("ALTER TABLE driver_runtime ADD COLUMN latitude DOUBLE PRECISION"))
            if "longitude" not in runtime_columns:
                conn.execute(text("ALTER TABLE driver_runtime ADD COLUMN longitude DOUBLE PRECISION"))

            if "location_source" not in violations_columns and inspector.has_table("driver_violations"):
                conn.execute(text("ALTER TABLE driver_violations ADD COLUMN location_source VARCHAR(16)"))
            if "event_codes" not in violations_columns and inspector.has_table("driver_violations"):
                conn.execute(text("ALTER TABLE driver_violations ADD COLUMN event_codes TEXT"))
            if "event_messages" not in violations_columns and inspector.has_table("driver_violations"):
                conn.execute(text("ALTER TABLE driver_violations ADD COLUMN event_messages TEXT"))

    @staticmethod
    def _serialize_json(values: list[str] | None) -> str | None:
        if values is None:
            return None
        return json.dumps(values)

    @staticmethod
    def _parse_json_list(raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            values = json.loads(raw)
            if isinstance(values, list):
                return [str(v) for v in values]
        except Exception:
            return []
        return []

    @staticmethod
    def _to_user_record(row: UserORM) -> UserRecord:
        return UserRecord(
            id=row.id,
            full_name=row.full_name,
            email=row.email,
            role=row.role,  # type: ignore[arg-type]
            password_hash=row.password_hash,
            phone_number=row.phone_number,
            license_plate=row.license_plate,
        )

    def create_user(
        self,
        full_name: str,
        email: str,
        role: UserRole,
        password_hash: str,
        phone_number: str | None = None,
        license_plate: str | None = None,
    ) -> UserRecord:
        normalized = email.lower().strip()
        clean_full_name = full_name.strip()
        clean_phone = (phone_number or "").strip()
        clean_plate = (license_plate or "").strip().upper()

        if not clean_full_name:
            raise ValueError("Họ tên không được để trống")
        if not normalized:
            raise ValueError("Email không được để trống")
        if role == "DRIVER" and not clean_phone:
            raise ValueError("Số điện thoại không được để trống")
        if role == "DRIVER" and not clean_plate:
            raise ValueError("Biển số xe không được để trống")

        user_id = str(uuid.uuid4())

        with self.SessionLocal() as db:
            row = UserORM(
                id=user_id,
                full_name=clean_full_name,
                email=normalized,
                role=role,
                password_hash=password_hash,
                phone_number=(clean_phone or None),
                license_plate=(clean_plate or None),
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError as ex:
                db.rollback()
                raise ValueError("Email đã tồn tại") from ex

            db.refresh(row)
            return self._to_user_record(row)

    def ensure_user(
        self,
        full_name: str,
        email: str,
        role: UserRole,
        password_hash: str,
        phone_number: str | None = None,
        license_plate: str | None = None,
    ) -> UserRecord:
        normalized = email.lower().strip()
        with self.SessionLocal() as db:
            row = db.scalar(select(UserORM).where(UserORM.email == normalized))
            if row is None:
                row = UserORM(
                    id=str(uuid.uuid4()),
                    full_name=full_name.strip(),
                    email=normalized,
                    role=role,
                    password_hash=password_hash,
                    phone_number=(phone_number or None),
                    license_plate=(license_plate or None),
                )
                db.add(row)
            else:
                row.full_name = full_name.strip()
                row.role = role
                row.password_hash = password_hash
                row.phone_number = (phone_number or row.phone_number)
                row.license_plate = (license_plate or row.license_plate)

            db.commit()
            db.refresh(row)
            return self._to_user_record(row)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        normalized = email.lower().strip()
        with self.SessionLocal() as db:
            row = db.scalar(select(UserORM).where(UserORM.email == normalized))
            if row is None:
                return None
            return self._to_user_record(row)

    def update_user_profile(
        self,
        user_id: str,
        full_name: str,
        email: str,
    ) -> UserRecord:
        clean_full_name = full_name.strip()
        normalized = email.lower().strip()
        if not clean_full_name:
            raise ValueError("Họ tên không được để trống")
        if not normalized:
            raise ValueError("Email không được để trống")

        with self.SessionLocal() as db:
            row = db.scalar(select(UserORM).where(UserORM.id == user_id))
            if row is None:
                raise ValueError("Tài khoản không tồn tại")

            row.full_name = clean_full_name
            row.email = normalized
            try:
                db.commit()
            except IntegrityError as ex:
                db.rollback()
                raise ValueError("Email đã tồn tại") from ex

            db.refresh(row)
            return self._to_user_record(row)

    def list_users(self) -> list[UserRecord]:
        with self.SessionLocal() as db:
            rows = db.scalars(select(UserORM).order_by(UserORM.email.asc())).all()
            return [self._to_user_record(r) for r in rows]

    def update_user_admin(
        self,
        *,
        user_id: str,
        full_name: str,
        email: str,
        role: UserRole,
        phone_number: str | None,
        license_plate: str | None,
    ) -> UserRecord:
        clean_full_name = full_name.strip()
        normalized = email.lower().strip()
        clean_phone = (phone_number or "").strip() or None
        clean_plate = (license_plate or "").strip().upper() or None

        if not clean_full_name:
            raise ValueError("Họ tên không được để trống")
        if not normalized:
            raise ValueError("Email không được để trống")

        with self.SessionLocal() as db:
            row = db.scalar(select(UserORM).where(UserORM.id == user_id))
            if row is None:
                raise ValueError("Tài khoản không tồn tại")

            row.full_name = clean_full_name
            row.email = normalized
            row.role = role
            row.phone_number = clean_phone
            row.license_plate = clean_plate

            try:
                db.commit()
            except IntegrityError as ex:
                db.rollback()
                raise ValueError("Email đã tồn tại") from ex

            db.refresh(row)
            return self._to_user_record(row)

    def delete_user_admin(self, user_id: str) -> bool:
        with self.SessionLocal() as db:
            row = db.scalar(select(UserORM).where(UserORM.id == user_id))
            if row is None:
                return False

            db.execute(text("DELETE FROM driver_runtime WHERE driver_id = :driver_id"), {"driver_id": user_id})
            db.execute(text("DELETE FROM driver_violations WHERE driver_id = :driver_id"), {"driver_id": user_id})
            db.delete(row)
            db.commit()
            return True

    def get_db_overview(self) -> tuple[str, list[tuple[str, int]]]:
        inspector = inspect(self.engine)
        # Keep this explicit to avoid querying system tables.
        table_names = [
            "users",
            "driver_runtime",
            "driver_violations",
            "vehicles",
            "trip_logs",
            "alert_history",
            "system_settings",
        ]

        rows: list[tuple[str, int]] = []
        with self.engine.begin() as conn:
            for name in table_names:
                if not inspector.has_table(name):
                    continue
                result = conn.execute(text(f"SELECT COUNT(*) AS total FROM {name}"))
                total = int(result.scalar_one() or 0)
                rows.append((name, total))

        masked = self.database_url
        if "@" in masked and "//" in masked:
            left, right = masked.split("//", 1)
            if "@" in right and ":" in right.split("@", 1)[0]:
                creds, tail = right.split("@", 1)
                user = creds.split(":", 1)[0]
                masked = f"{left}//{user}:***@{tail}"

        return masked, rows

    def update_driver_runtime(
        self,
        driver_user: UserRecord,
        risk_score: int,
        risk_level: RiskLevel,
        speed_kmh: float,
        latitude: float | None,
        longitude: float | None,
        frame_base64: str | None,
    ) -> None:
        with self.SessionLocal() as db:
            now = int(time.time() * 1000)
            item = db.scalar(select(DriverRuntimeORM).where(DriverRuntimeORM.driver_id == driver_user.id))
            if item is None:
                item = DriverRuntimeORM(
                    driver_id=driver_user.id,
                    driver_name=driver_user.full_name,
                    email=driver_user.email,
                    phone_number=driver_user.phone_number,
                    license_plate=driver_user.license_plate,
                )
                db.add(item)

            item.driver_name = driver_user.full_name
            item.email = driver_user.email
            item.phone_number = driver_user.phone_number
            item.license_plate = driver_user.license_plate
            item.risk_score = risk_score
            item.risk_level = risk_level
            item.speed_kmh = speed_kmh
            item.latitude = latitude
            item.longitude = longitude
            item.last_seen_unix_ms = now
            if frame_base64:
                item.latest_frame_base64 = frame_base64

            db.commit()

    def list_driver_runtime(self) -> list[DriverRuntime]:
        with self.SessionLocal() as db:
            rows = db.scalars(select(DriverRuntimeORM)).all()
            return [
                DriverRuntime(
                    driver_id=r.driver_id,
                    driver_name=r.driver_name,
                    email=r.email,
                    phone_number=r.phone_number,
                    license_plate=r.license_plate,
                    risk_score=r.risk_score,
                    risk_level=r.risk_level,  # type: ignore[arg-type]
                    speed_kmh=r.speed_kmh,
                    latitude=r.latitude,
                    longitude=r.longitude,
                    last_seen_unix_ms=r.last_seen_unix_ms,
                    latest_frame_base64=r.latest_frame_base64,
                )
                for r in rows
            ]

    def create_driver_violation_if_needed(
        self,
        *,
        driver_user: UserRecord,
        risk_score: int,
        risk_level: RiskLevel,
        threshold: int,
        cooldown_sec: int,
        latitude: float | None,
        longitude: float | None,
        location_source: str | None,
        frame_base64: str | None,
        event_codes: list[str],
        event_messages: list[str],
    ) -> bool:
        if risk_level != "CRITICAL" or risk_score < threshold or not frame_base64:
            return False

        now_ms = int(time.time() * 1000)
        cooldown_ms = max(0, cooldown_sec) * 1000

        with self.SessionLocal() as db:
            last_item = db.scalars(
                select(DriverViolationORM)
                .where(DriverViolationORM.driver_id == driver_user.id)
                .order_by(DriverViolationORM.created_at_unix_ms.desc())
                .limit(1)
            ).first()

            if last_item is not None and (now_ms - int(last_item.created_at_unix_ms)) < cooldown_ms:
                return False

            item = DriverViolationORM(
                driver_id=driver_user.id,
                risk_score=risk_score,
                risk_level=risk_level,
                latitude=latitude,
                longitude=longitude,
                location_source=location_source,
                event_codes=self._serialize_json(event_codes),
                event_messages=self._serialize_json(event_messages),
                frame_base64=frame_base64,
                created_at_unix_ms=now_ms,
            )
            db.add(item)
            db.commit()
            return True

    def list_driver_violations(self, driver_id: str, limit: int = 50) -> list[DriverViolation]:
        safe_limit = max(1, min(200, limit))
        with self.SessionLocal() as db:
            rows = db.scalars(
                select(DriverViolationORM)
                .where(DriverViolationORM.driver_id == driver_id)
                .order_by(DriverViolationORM.created_at_unix_ms.desc())
                .limit(safe_limit)
            ).all()

            return [
                DriverViolation(
                    id=r.id,
                    driver_id=r.driver_id,
                    risk_score=r.risk_score,
                    risk_level=r.risk_level,  # type: ignore[arg-type]
                    latitude=r.latitude,
                    longitude=r.longitude,
                    location_source=r.location_source,
                    event_codes=self._parse_json_list(r.event_codes),
                    event_messages=self._parse_json_list(r.event_messages),
                    frame_base64=r.frame_base64,
                    created_at_unix_ms=r.created_at_unix_ms,
                )
                for r in rows
            ]


store = PostgresStore()
