import tempfile
import unittest
from pathlib import Path

from core.datasets import read_detection_events


class DatasetTests(unittest.TestCase):
    def test_prediction_reader_preserves_scores_for_calibration(self):
        columns = [f"prob_S{index}" for index in range(1, 17)]
        values = ["0.8"] + ["0.01"] * 15
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                f"time_sec,predicted_strings,{','.join(columns)}\n"
                f"1.25,1,{','.join(values)}\n",
                encoding="utf-8",
            )
            events = read_detection_events(path, require_scores=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].strings, frozenset({1}))
        self.assertEqual(events[0].scores[0], 0.8)

    def test_partial_probability_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("time_sec,strings,prob_S1\n1.0,1,0.8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "part of the prob_S1"):
                read_detection_events(path)


if __name__ == "__main__":
    unittest.main()
