from __future__ import annotations

from threading import RLock
from typing import Any


class JobStore:
    """Small thread-safe store for request and background-task job state."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._jobs

    def __getitem__(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._jobs[job_id])

    def __setitem__(self, job_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = dict(state)

    def get(self, job_id: str, default: Any = None) -> dict[str, Any] | Any:
        with self._lock:
            state = self._jobs.get(job_id)
            return dict(state) if state is not None else default

    def set_stage(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        message: str,
        progress: int,
        **result: Any,
    ) -> None:
        normalized_progress = max(0, min(100, int(progress)))
        with self._lock:
            self._jobs[job_id] = {
                "status": status,
                "stage": stage,
                "message": message,
                "progress": normalized_progress,
                **result,
            }
