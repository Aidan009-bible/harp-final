import json
import tempfile
import unittest
from pathlib import Path

from core.calibration import load_thresholds, tune_per_string_thresholds
from core.metrics import DetectionEvent


def scores(**overrides: float) -> tuple[float, ...]:
    values = [0.01] * 16
    for label, value in overrides.items():
        values[int(label.removeprefix("S")) - 1] = value
    return tuple(values)


class CalibrationTests(unittest.TestCase):
    def test_tuning_finds_a_per_string_validation_operating_point(self):
        predictions = [
            DetectionEvent(1.00, frozenset({1}), scores(S1=0.80)),
            DetectionEvent(2.00, frozenset({1}), scores(S1=0.40)),
        ]
        truth = [DetectionEvent(1.01, frozenset({1}))]

        report = tune_per_string_thresholds(
            predictions,
            truth,
            tolerance_sec=0.05,
            minimum_support=1,
        )

        self.assertEqual(report["thresholds"]["S1"], 0.8)
        self.assertEqual(report["per_string"]["S1"]["tp"], 1)
        self.assertEqual(report["per_string"]["S1"]["fp"], 0)
        self.assertEqual(report["per_string"]["S2"]["support"], 0)
        self.assertIn("S1", report["calibrated_strings"])
        self.assertIn("S2", report["fallback_strings"])

    def test_load_thresholds_requires_all_valid_string_values(self):
        payload = {"thresholds": {f"S{index}": 0.2 for index in range(1, 17)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_thresholds(path)

        self.assertEqual(loaded, (0.2,) * 16)


if __name__ == "__main__":
    unittest.main()
