from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Protocol


UPLOAD_CHUNK_BYTES = 1024 * 1024


class ReadableUpload(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


class UploadValidationError(ValueError):
    """Raised when an uploaded file does not meet the public API contract."""


def safe_upload_name(
    filename: str | None,
    *,
    fallback: str,
    allowed_suffixes: Iterable[str],
) -> str:
    raw_name = (filename or fallback).replace("\\", "/").split("/")[-1]
    cleaned_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
    cleaned_name = cleaned_name[:120] or fallback
    suffix = Path(cleaned_name).suffix.lower()
    allowed = {item.lower() for item in allowed_suffixes}
    if suffix not in allowed:
        expected = ", ".join(sorted(allowed))
        raise UploadValidationError(f"Unsupported file type. Expected one of: {expected}")
    return cleaned_name


async def save_upload(
    upload: ReadableUpload,
    directory: Path,
    *,
    fallback: str,
    allowed_suffixes: Iterable[str],
    max_bytes: int,
) -> Path:
    filename = safe_upload_name(
        upload.filename,
        fallback=fallback,
        allowed_suffixes=allowed_suffixes,
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    bytes_written = 0

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise UploadValidationError(
                        f"{filename} is larger than the {max_bytes // (1024 * 1024)} MB limit."
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise UploadValidationError(f"{filename} is empty.")

    return destination
