from __future__ import annotations

import csv
import math
from pathlib import Path

from core.metrics import DetectionEvent, parse_string_set


NUM_STRINGS = 16


def read_detection_events(path: Path, *, require_scores: bool = False) -> list[DetectionEvent]:
    """Read prediction or ground-truth events from the project's CSV format."""
    events: list[DetectionEvent] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")

        score_columns = [f"prob_S{string_id}" for string_id in range(1, NUM_STRINGS + 1)]
        present_score_columns = [column for column in score_columns if column in reader.fieldnames]
        if present_score_columns and len(present_score_columns) != NUM_STRINGS:
            raise ValueError(f"{path} contains only part of the prob_S1..prob_S16 score set")
        has_scores = len(present_score_columns) == NUM_STRINGS
        if require_scores and not has_scores:
            raise ValueError(f"{path} must contain prob_S1..prob_S16 for threshold tuning")

        for row_number, row in enumerate(reader, start=2):
            try:
                time_sec = float(row["time_sec"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number} has an invalid time_sec") from error
            if not math.isfinite(time_sec) or time_sec < 0:
                raise ValueError(f"{path}:{row_number} has an invalid time_sec")

            strings = parse_string_set(row.get("predicted_strings") or row.get("strings") or "")
            scores = None
            if has_scores:
                try:
                    parsed_scores = tuple(float(row[column]) for column in score_columns)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{row_number} has an invalid probability") from error
                if any(not math.isfinite(score) or not 0 <= score <= 1 for score in parsed_scores):
                    raise ValueError(
                        f"{path}:{row_number} probabilities must be finite values between 0 and 1"
                    )
                scores = parsed_scores

            events.append(DetectionEvent(time_sec=time_sec, strings=strings, scores=scores))
    return events
