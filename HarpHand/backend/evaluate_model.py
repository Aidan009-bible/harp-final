"""Evaluate HarpHand prediction CSVs against human-labeled events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.datasets import read_detection_events
from core.metrics import evaluate_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prediction events with a labeled CSV using a timestamp tolerance.",
    )
    parser.add_argument("predictions", type=Path, help="CSV with time_sec and predicted_strings")
    parser.add_argument("ground_truth", type=Path, help="CSV with time_sec and strings")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Event matching tolerance in seconds")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if args.tolerance < 0:
        parser.error("--tolerance must be zero or greater")

    report = evaluate_events(
        read_detection_events(args.predictions),
        read_detection_events(args.ground_truth),
        tolerance_sec=args.tolerance,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
