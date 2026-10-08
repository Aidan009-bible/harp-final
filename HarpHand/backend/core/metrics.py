from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DetectionEvent:
    time_sec: float
    strings: frozenset[int]
    scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class _AlignmentNode:
    prediction_index: int
    truth_index: int
    previous: _AlignmentNode | None


@dataclass(frozen=True)
class _AlignmentState:
    matches: int = 0
    timing_error: float = 0.0
    node: _AlignmentNode | None = None


def _better_alignment(candidate: _AlignmentState, current: _AlignmentState) -> bool:
    if candidate.matches != current.matches:
        return candidate.matches > current.matches
    return candidate.timing_error < current.timing_error


def parse_string_set(value: str | Iterable[int]) -> frozenset[int]:
    if isinstance(value, str):
        raw_values = value.replace("S", "").replace("s", "").split(",")
    else:
        raw_values = value

    strings = set()
    for raw_value in raw_values:
        try:
            string_id = int(str(raw_value).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= string_id <= 16:
            strings.add(string_id)
    return frozenset(strings)


def match_events(
    predictions: list[DetectionEvent],
    ground_truth: list[DetectionEvent],
    tolerance_sec: float,
) -> tuple[list[tuple[DetectionEvent, DetectionEvent]], list[DetectionEvent], list[DetectionEvent]]:
    """Align ordered events one-to-one, maximizing matches then minimizing timing error."""
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec must be zero or greater")

    ordered_predictions = sorted(predictions, key=lambda event: event.time_sec)
    ordered_truth = sorted(ground_truth, key=lambda event: event.time_sec)
    truth_times = [event.time_sec for event in ordered_truth]

    # Fenwick tree over truth indices. Candidate edges exist only inside the
    # tolerance window, so memory scales with plausible matches rather than
    # len(predictions) * len(ground_truth).
    tree = [_AlignmentState() for _ in range(len(ordered_truth) + 1)]

    def query(prefix_length: int) -> _AlignmentState:
        best = _AlignmentState()
        index = prefix_length
        while index > 0:
            if _better_alignment(tree[index], best):
                best = tree[index]
            index -= index & -index
        return best

    def update(position: int, state: _AlignmentState) -> None:
        index = position
        while index < len(tree):
            if _better_alignment(state, tree[index]):
                tree[index] = state
            index += index & -index

    for prediction_index, prediction in enumerate(ordered_predictions):
        lower = bisect_left(truth_times, prediction.time_sec - tolerance_sec)
        upper = bisect_right(truth_times, prediction.time_sec + tolerance_sec)
        pending_updates: list[tuple[int, _AlignmentState]] = []
        for truth_index in range(lower, upper):
            previous = query(truth_index)
            timing_error = abs(prediction.time_sec - ordered_truth[truth_index].time_sec)
            pending_updates.append(
                (
                    truth_index + 1,
                    _AlignmentState(
                        matches=previous.matches + 1,
                        timing_error=previous.timing_error + timing_error,
                        node=_AlignmentNode(
                            prediction_index=prediction_index,
                            truth_index=truth_index,
                            previous=previous.node,
                        ),
                    ),
                )
            )
        # Delay updates until every edge for this prediction is scored, which
        # prevents one prediction event from being matched more than once.
        for position, state in pending_updates:
            update(position, state)

    best = query(len(ordered_truth))
    matched_indices: list[tuple[int, int]] = []
    node = best.node
    while node is not None:
        matched_indices.append((node.prediction_index, node.truth_index))
        node = node.previous
    matched_indices.reverse()

    matched_prediction_indices = {prediction_index for prediction_index, _ in matched_indices}
    matched_truth_indices = {truth_index for _, truth_index in matched_indices}
    matches = [
        (ordered_predictions[prediction_index], ordered_truth[truth_index])
        for prediction_index, truth_index in matched_indices
    ]
    unmatched_predictions = [
        prediction
        for index, prediction in enumerate(ordered_predictions)
        if index not in matched_prediction_indices
    ]
    unmatched_truth = [
        truth
        for index, truth in enumerate(ordered_truth)
        if index not in matched_truth_indices
    ]
    return matches, unmatched_predictions, unmatched_truth


def evaluate_events(
    predictions: list[DetectionEvent],
    ground_truth: list[DetectionEvent],
    tolerance_sec: float = 0.15,
) -> dict:
    matches, unmatched_predictions, unmatched_truth = match_events(
        predictions,
        ground_truth,
        tolerance_sec,
    )
    event_precision = len(matches) / len(predictions) if predictions else 0.0
    event_recall = len(matches) / len(ground_truth) if ground_truth else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )

    label_tp = label_fp = label_fn = exact_sets = 0
    per_string = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    timing_errors = []
    matched_brier_terms: list[float] = []
    pipeline_brier_terms: list[float] = []
    scores_available = bool(predictions) and all(
        prediction.scores is not None and len(prediction.scores) == 16
        for prediction in predictions
    )

    for prediction, truth in matches:
        timing_errors.append(abs(prediction.time_sec - truth.time_sec))
        exact_sets += prediction.strings == truth.strings
        for string_id in range(1, 17):
            predicted = string_id in prediction.strings
            expected = string_id in truth.strings
            if predicted and expected:
                label_tp += 1
                per_string[string_id]["tp"] += 1
            elif predicted:
                label_fp += 1
                per_string[string_id]["fp"] += 1
            elif expected:
                label_fn += 1
                per_string[string_id]["fn"] += 1
            if scores_available:
                squared_error = (float(prediction.scores[string_id - 1]) - float(expected)) ** 2
                matched_brier_terms.append(squared_error)
                pipeline_brier_terms.append(squared_error)

    for prediction in unmatched_predictions:
        for string_id in prediction.strings:
            label_fp += 1
            per_string[string_id]["fp"] += 1
        if scores_available:
            pipeline_brier_terms.extend(float(score) ** 2 for score in prediction.scores)
    for truth in unmatched_truth:
        for string_id in truth.strings:
            label_fn += 1
            per_string[string_id]["fn"] += 1
        if scores_available:
            pipeline_brier_terms.extend(
                1.0 if string_id in truth.strings else 0.0
                for string_id in range(1, 17)
            )

    label_precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
    label_recall = label_tp / (label_tp + label_fn) if label_tp + label_fn else 0.0
    label_f1 = (
        2 * label_precision * label_recall / (label_precision + label_recall)
        if label_precision + label_recall
        else 0.0
    )

    per_string_metrics = {}
    for string_id in range(1, 17):
        counts = per_string[string_id]
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_string_metrics[f"S{string_id}"] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "tolerance_sec": tolerance_sec,
        "prediction_events": len(predictions),
        "truth_events": len(ground_truth),
        "matched_events": len(matches),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "exact_string_set_rate": exact_sets / len(matches) if matches else 0.0,
        "label_micro_precision": label_precision,
        "label_micro_recall": label_recall,
        "label_micro_f1": label_f1,
        "mean_timing_error_sec": sum(timing_errors) / len(timing_errors) if timing_errors else None,
        "label_brier_score_matched_events": (
            sum(matched_brier_terms) / len(matched_brier_terms)
            if matched_brier_terms
            else None
        ),
        "label_brier_score_pipeline": (
            sum(pipeline_brier_terms) / len(pipeline_brier_terms)
            if pipeline_brier_terms
            else None
        ),
        "per_string": per_string_metrics,
    }
