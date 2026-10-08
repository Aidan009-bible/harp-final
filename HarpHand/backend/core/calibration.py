from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.metrics import DetectionEvent, match_events


NUM_STRINGS = 16


def _classification_metrics(
    samples: Iterable[tuple[float, bool]],
    *,
    threshold: float,
    missing_positives: int = 0,
) -> dict[str, float | int]:
    tp = fp = 0
    fn = int(missing_positives)
    for score, expected in samples:
        predicted = score >= threshold
        if predicted and expected:
            tp += 1
        elif predicted:
            fp += 1
        elif expected:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_per_string_thresholds(
    predictions: list[DetectionEvent],
    ground_truth: list[DetectionEvent],
    *,
    tolerance_sec: float = 0.15,
    minimum_support: int = 5,
    fallback_thresholds: Iterable[float] = (0.25,) * NUM_STRINGS,
) -> dict:
    """Select one validation-set F1 threshold per string after temporal alignment."""
    if minimum_support < 1:
        raise ValueError("minimum_support must be at least 1")
    if any(event.scores is None or len(event.scores) != NUM_STRINGS for event in predictions):
        raise ValueError("Every prediction event must contain 16 probability scores")
    fallback_values = tuple(float(value) for value in fallback_thresholds)
    if len(fallback_values) != NUM_STRINGS or any(
        not 0 < value <= 1 for value in fallback_values
    ):
        raise ValueError("fallback_thresholds must contain 16 values in the interval (0, 1]")

    matches, unmatched_predictions, unmatched_truth = match_events(
        predictions,
        ground_truth,
        tolerance_sec,
    )
    thresholds: dict[str, float] = {}
    per_string: dict[str, dict] = {}
    supported_f1: list[float] = []
    calibrated_strings: list[str] = []
    fallback_strings: list[str] = []

    for string_id in range(1, NUM_STRINGS + 1):
        score_index = string_id - 1
        samples: list[tuple[float, bool]] = [
            (float(prediction.scores[score_index]), string_id in truth.strings)
            for prediction, truth in matches
        ]
        samples.extend(
            (float(prediction.scores[score_index]), False)
            for prediction in unmatched_predictions
        )
        missing_positives = sum(string_id in truth.strings for truth in unmatched_truth)
        support = sum(expected for _, expected in samples) + missing_positives

        label = f"S{string_id}"
        if support < minimum_support:
            best_threshold = fallback_values[score_index]
            best_metrics = _classification_metrics(
                samples,
                threshold=best_threshold,
                missing_positives=missing_positives,
            )
            threshold_source = "fallback_insufficient_support"
            fallback_strings.append(label)
        else:
            candidates = sorted(
                {round(score, 8) for score, _ in samples if score > 0} | {0.01, 1.0}
            )
            scored_candidates = [
                (
                    threshold,
                    _classification_metrics(
                        samples,
                        threshold=threshold,
                        missing_positives=missing_positives,
                    ),
                )
                for threshold in candidates
            ]
            best_threshold, best_metrics = max(
                scored_candidates,
                key=lambda item: (
                    item[1]["f1"],
                    item[1]["precision"],
                    item[0],
                ),
            )
            supported_f1.append(float(best_metrics["f1"]))
            threshold_source = "validation"
            calibrated_strings.append(label)

        thresholds[label] = best_threshold
        per_string[label] = {
            "threshold": best_threshold,
            "threshold_source": threshold_source,
            "support": support,
            **best_metrics,
        }

    return {
        "schema_version": 1,
        "selection_split": "validation",
        "selection_metric": "per_string_f1",
        "minimum_support": minimum_support,
        "tolerance_sec": tolerance_sec,
        "matched_events": len(matches),
        "unmatched_prediction_events": len(unmatched_predictions),
        "unmatched_truth_events": len(unmatched_truth),
        "macro_f1_supported_strings": (
            sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0
        ),
        "calibrated_strings": calibrated_strings,
        "fallback_strings": fallback_strings,
        "thresholds": thresholds,
        "per_string": per_string,
    }


def load_thresholds(path: Path) -> tuple[float, ...]:
    """Load a threshold report and validate all 16 operating points."""
    if not path.exists():
        raise ValueError(f"Threshold file does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read threshold file: {path}") from error
    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("Threshold file must contain a 'thresholds' object")

    values = []
    for string_id in range(1, NUM_STRINGS + 1):
        label = f"S{string_id}"
        try:
            value = float(raw_thresholds[label])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Threshold file has no valid value for {label}") from error
        if not 0 < value <= 1:
            raise ValueError(f"Threshold for {label} must be greater than 0 and at most 1")
        values.append(value)
    return tuple(values)
