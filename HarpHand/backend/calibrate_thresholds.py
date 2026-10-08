"""Tune per-string inference thresholds on a labeled validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.calibration import tune_per_string_thresholds
from core.datasets import read_detection_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune 16 per-string thresholds from validation predictions and labels.",
    )
    parser.add_argument("predictions", type=Path, help="CSV with time_sec and prob_S1..prob_S16")
    parser.add_argument("ground_truth", type=Path, help="CSV with time_sec and strings")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Event matching tolerance in seconds")
    parser.add_argument(
        "--minimum-support",
        type=int,
        default=5,
        help="Minimum positive validation events required to tune a string",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON threshold report path")
    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance must be zero or greater")
    if args.minimum_support < 1:
        parser.error("--minimum-support must be at least 1")

    report = tune_per_string_thresholds(
        read_detection_events(args.predictions, require_scores=True),
        read_detection_events(args.ground_truth),
        tolerance_sec=args.tolerance,
        minimum_support=args.minimum_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
