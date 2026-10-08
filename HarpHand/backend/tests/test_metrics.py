import unittest

from core.metrics import DetectionEvent, evaluate_events, match_events, parse_string_set


class MetricsTests(unittest.TestCase):
    def test_match_events_maximizes_cardinality_for_dense_plucks(self):
        predictions = [
            DetectionEvent(0.15, frozenset({1})),
            DetectionEvent(0.17, frozenset({2})),
        ]
        truth = [
            DetectionEvent(0.00, frozenset({1})),
            DetectionEvent(0.16, frozenset({2})),
        ]

        matches, unmatched_predictions, unmatched_truth = match_events(
            predictions,
            truth,
            tolerance_sec=0.15,
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_predictions, [])
        self.assertEqual(unmatched_truth, [])

    def test_match_events_minimizes_timing_error_after_match_count(self):
        predictions = [
            DetectionEvent(0.14, frozenset({1})),
            DetectionEvent(0.29, frozenset({2})),
        ]
        truth = [
            DetectionEvent(0.00, frozenset({3})),
            DetectionEvent(0.15, frozenset({1})),
            DetectionEvent(0.30, frozenset({2})),
        ]

        matches, _, unmatched_truth = match_events(predictions, truth, tolerance_sec=0.15)

        self.assertEqual([(prediction.time_sec, expected.time_sec) for prediction, expected in matches], [
            (0.14, 0.15),
            (0.29, 0.30),
        ])
        self.assertEqual([event.time_sec for event in unmatched_truth], [0.00])

    def test_parse_string_set_accepts_exported_labels(self):
        self.assertEqual(parse_string_set("S1, 2, bad, 17"), frozenset({1, 2}))

    def test_evaluate_events_reports_timing_and_label_quality(self):
        predictions = [
            DetectionEvent(1.05, frozenset({1, 2})),
            DetectionEvent(3.00, frozenset({4})),
        ]
        truth = [
            DetectionEvent(1.00, frozenset({1, 3})),
            DetectionEvent(2.00, frozenset({4})),
        ]

        report = evaluate_events(predictions, truth, tolerance_sec=0.10)

        self.assertEqual(report["matched_events"], 1)
        self.assertAlmostEqual(report["event_precision"], 0.5)
        self.assertAlmostEqual(report["event_recall"], 0.5)
        self.assertAlmostEqual(report["mean_timing_error_sec"], 0.05)
        self.assertEqual(report["per_string"]["S1"]["tp"], 1)
        self.assertEqual(report["per_string"]["S2"]["fp"], 1)
        self.assertEqual(report["per_string"]["S3"]["fn"], 1)

    def test_evaluate_events_reports_probability_quality_when_scores_exist(self):
        prediction_scores = (0.8,) + (0.0,) * 15
        report = evaluate_events(
            [DetectionEvent(1.0, frozenset({1}), prediction_scores)],
            [DetectionEvent(1.0, frozenset({1}))],
        )

        self.assertAlmostEqual(report["label_brier_score_matched_events"], 0.0025)
        self.assertAlmostEqual(report["label_brier_score_pipeline"], 0.0025)


if __name__ == "__main__":
    unittest.main()
