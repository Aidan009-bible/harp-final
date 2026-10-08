# Model evaluation protocol

The bundled audio model is a 16-label Keras classifier with two inputs: a `128 × 51 × 1` mel-spectrogram and a 16-value harmonic-energy vector. Its output is a 16-value sigmoid vector. The repository does not include the training dataset, held-out split, calibration report, or performer-level benchmark, so the model's real-world accuracy cannot be established from the artifact alone.

## Evaluation data

Create a ground-truth CSV for recordings that were not used during training. Split by performer and recording session—not by random audio clips from the same recording—to reduce leakage.

```csv
time_sec,strings
0.84,"4"
1.27,"4,9"
2.11,"12"
```

Use diverse instruments, performers, camera angles, rooms, microphones, tempos, dynamics, and single- versus multi-string events. Keep a frozen test set that is not used for threshold tuning.

## Run the evaluator

Generate a prediction CSV through HarpHand, then run:

```bash
cd backend
python evaluate_model.py outputs/<job-id>/predictions_hybrid.csv ground_truth.csv \
  --tolerance 0.15 \
  --output evaluation-report.json
```

The report separates:

- event precision, recall, and F1: whether pluck times were found;
- exact string-set rate: whether every label at a matched event was correct;
- micro label precision, recall, and F1;
- per-string precision, recall, and F1;
- mean absolute timing error.
- Brier scores for matched events and the complete event pipeline when probability columns are available.

## Tune validation thresholds

Run inference on a validation split that is separate from the frozen test set. The prediction CSV already contains `prob_S1` through `prob_S16`:

```bash
cd backend
python calibrate_thresholds.py validation_predictions.csv validation_ground_truth.csv \
  --tolerance 0.15 \
  --minimum-support 5 \
  --output calibration/thresholds.json
```

Strings below the minimum positive-event support retain the baseline threshold and are listed under `fallback_strings` in the report. Raise the minimum for a serious benchmark; five is only a guardrail for an early prototype.

Start the API with `HARP_THRESHOLDS_PATH` pointing to that JSON file. Each audio run writes `inference_manifest.json` with the model checksum, selected thresholds, preprocessing constants, and relevant library versions.

Do not select thresholds on the frozen test split. This would leak test labels into the decision rule and make the final score optimistic.

Audio/hand agreement in the UI is a diagnostic comparison between two detectors. It is not a substitute for ground-truth accuracy.

## Decision gate for retraining

Retrain only after the frozen benchmark identifies a material gap, such as poor recall on specific strings, weak multi-string recognition, or domain shift across instruments and rooms. Compare the current artifact and each candidate model on the same frozen test set. Record the dataset version, code revision, thresholds, metrics, and model checksum with every result.

Before replacing the bundled model, require:

1. no regression in event precision and recall;
2. improved macro/per-string F1 for the targeted failure modes;
3. an explicit check for performer and session leakage;
4. calibrated thresholds selected on validation data, never on the frozen test set;
5. a manual review of representative false positives and false negatives.

See [the research roadmap](research-roadmap.md) for the dataset design and staged improvement plan.
