from __future__ import annotations

import os
from dataclasses import dataclass


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    allow_custom_model_uploads: bool
    max_video_bytes: int
    max_model_bytes: int
    max_weights_bytes: int
    hand_pre_onset_sec: float
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        default_origins = (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://harp-final.vercel.app",
        )
        configured_origins = tuple(
            origin.strip()
            for origin in os.getenv("HARP_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        )
        megabyte = 1024 * 1024
        return cls(
            allow_custom_model_uploads=env_flag("ALLOW_CUSTOM_MODEL_UPLOADS", False),
            max_video_bytes=env_int("MAX_VIDEO_MB", 250, 1) * megabyte,
            max_model_bytes=env_int("MAX_MODEL_MB", 200, 1) * megabyte,
            max_weights_bytes=env_int("MAX_WEIGHTS_MB", 500, 1) * megabyte,
            hand_pre_onset_sec=env_int("HAND_PRE_ONSET_MS", 150, 10) / 1000.0,
            cors_origins=configured_origins or default_origins,
        )
