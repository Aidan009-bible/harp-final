# HarpHand research roadmap

This roadmap separates improvements that can be justified with the current repository from changes that require new labeled data. It is a selected, non-exhaustive evidence pass; the scholarly fan-out search could not run because no contact email was configured for its polite API identity.

## What the current artifact can and cannot prove

The bundled model is a 16-output sigmoid classifier, but the repository does not include its training split, performer/session identities, validation predictions, calibration report, or frozen test results. Its probabilities therefore should be treated as model scores, not verified confidence estimates. Audio/hand agreement is also not ground-truth accuracy.

## Implemented priorities

### 1. Calibrate before retraining

Neural-network probability estimates may be miscalibrated even when class predictions are useful. Decision thresholds also change the precision/recall operating point without changing the underlying model. HarpHand now supports:

- per-string threshold selection on a validation set;
- Brier scores for probability-quality diagnosis;
- loading the selected threshold profile through `HARP_THRESHOLDS_PATH`;
- an inference manifest containing the model SHA-256, input/output shapes, thresholds, and runtime versions.

Thresholds must be selected on validation data. The frozen test set is used once for the final comparison, never for threshold selection.

### 2. Make visual proximity resolution-independent

A fixed 20-pixel contact boundary changes meaning when video resolution changes. Hand contact now scales from the shorter frame dimension with practical minimum and maximum bounds. The detector records the threshold and normalized distance beside each touch event so results remain auditable.

MediaPipe is now used in video mode for decoded video frames. This makes the tracking-confidence setting operational and preserves timestamped temporal tracking instead of treating every frame as an unrelated image.

### 3. Keep event detection, label classification, and fusion separate

Onset detection is an upstream task. A missing onset cannot be repaired by lowering a string-class threshold, and a false onset should not be counted only as a label error. Reports therefore separate event precision/recall/F1, label metrics, timing error, and probability quality.

The audio and visual branches remain independently inspectable. Fusion timing is configurable through `HAND_PRE_ONSET_MS`; agreement remains a diagnostic, not an accuracy claim.

## Dataset and experiment plan

1. Record at least three performers across multiple sessions, instruments, rooms, microphones, camera positions, dynamics, and tempi.
2. Label onset time and the complete set of strings for every event. Keep performer and session identifiers in metadata.
3. Split by performer/session, not random clips from the same recording.
4. Use the training split for model fitting, validation for threshold/onset/fusion tuning, and one frozen test split for final reporting.
5. Report event F1, exact string-set rate, micro and per-string F1, Brier score, and timing error. Include representative false positives and false negatives.
6. Compare the bundled model, calibrated bundled model, and any retrained candidate on identical splits and inference code.

## When a new model is justified

Retraining is justified only if the calibrated baseline still shows a repeatable gap, such as:

- low recall concentrated on specific strings;
- poor multi-string event recognition;
- a clear performer, instrument, or room domain shift;
- probability quality too poor for stable threshold selection;
- audio errors that the independently evaluated visual branch consistently resolves.

If data volume is limited, first test augmentation or synthetic pretraining followed by real-data fine-tuning. Work on guitar tablature transcription suggests that cross-domain generalization is often limited by dataset size and diversity, so a larger architecture alone is not a serious research plan.

## Selected sources

- Guo, Pleiss, Sun, and Weinberger, [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017.
- Müller and Chiu, [A Basic Tutorial on Novelty and Activation Functions for Music Signal Processing](https://doi.org/10.5334/tismir.202), TISMIR 2024.
- Xi et al., [GuitarSet: A Dataset for Guitar Transcription](https://archives.ismir.net/ismir2018/paper/000188.pdf), ISMIR 2018.
- Zang et al., [SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription](https://arxiv.org/abs/2309.09085), 2023.
- Google AI Edge, [MediaPipe Hand Landmarker options](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions).
- librosa, [Onset detection documentation](https://librosa.org/doc/0.10.2/onset.html).
