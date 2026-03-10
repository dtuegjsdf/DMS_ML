import os
from pathlib import Path

from pydantic import BaseModel
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class AppConfig(BaseModel):
    app_name: str = "DMS Backend"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    hidden_ai_enabled: bool = os.getenv("HIDDEN_AI_ENABLED", "false").lower() == "true"
    hidden_ai_url: str = os.getenv("HIDDEN_AI_URL", "")
    hidden_ai_api_key: str = os.getenv("HIDDEN_AI_API_KEY", "")
    hidden_ai_model: str = os.getenv("HIDDEN_AI_MODEL", "gpt-4o-mini")
    hidden_ai_timeout_sec: float = float(os.getenv("HIDDEN_AI_TIMEOUT_SEC", "15"))
    l2cs_enabled: bool = os.getenv("L2CS_ENABLED", "false").lower() == "true"
    l2cs_adapter_module: str = os.getenv("L2CS_ADAPTER_MODULE", "")
    l2cs_adapter_function: str = os.getenv("L2CS_ADAPTER_FUNCTION", "infer_gaze")
    critical_violation_threshold: int = int(os.getenv("CRITICAL_VIOLATION_THRESHOLD", "70"))
    critical_violation_cooldown_sec: int = int(os.getenv("CRITICAL_VIOLATION_COOLDOWN_SEC", "8"))


config = AppConfig()
