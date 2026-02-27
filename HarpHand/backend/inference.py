# Harp string detection inference (hybrid model + YIN fallback)
# Adapted from Colab pipeline for local/API use.

import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

# Resolve ffmpeg to full path so subprocess finds it (Windows often misses PATH in child processes)
def _resolve_ffmpeg():
    _dir = Path(__file__).resolve().parent / "tools" / "ffmpeg"
    _exe = _dir / "bin" / "ffmpeg.exe"
    if _exe.exists():
        return str(_exe)
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common winget/install locations
    for prefix in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "ffmpeg",
    ]:
        if prefix and (prefix / "ffmpeg.exe").exists():
            return str(prefix / "ffmpeg.exe")
    return "ffmpeg"  # last resort; will raise WinError 2 with clear message below

FFMPEG_CMD = _resolve_ffmpeg()
import librosa
import pandas as pd
import tensorflow as tf

NUM_STRINGS = 16
SAMPLE_RATE = 16000
CLIP_SEC = 0.8
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SEC)
N_MELS = 128
N_FFT = 1024
HOP = 256
F_NFFT = 4096
F_HOP = 256

# Default: single threshold for all strings (first Colab way)
THR_DEFAULT = 0.25
# Hybrid: per-string thresholds (string 12 fix)
THR_ARRAY = np.full(NUM_STRINGS, 0.25, dtype=np.float32)
THR_ARRAY[11] = 0.03

MODEL_CONF_MIN = 0.20
# Include runner-up string in predicted_strings when above this (so grid can show e.g. "4,9")
TOP2_DISPLAY_THR = 0.20
YIN_WIN_SEC = 0.15
OFFSET_SEC = 0.03
HOLD_SEC = 0.18
FLASH_DURATION = 0.25

HARP_STRINGS = {
    1: 783.99, 2: 659.25, 3: 587.33, 4: 523.25, 5: 440.00, 6: 392.00,
    7: 329.63, 8: 293.66, 9: 261.63, 10: 220.00, 11: 196.00, 12: 164.81,
    13: 146.83, 14: 130.81, 15: 110.00, 16: 98.00,
}
FREQS = np.array([HARP_STRINGS[i] for i in range(1, 17)], dtype=np.float32)


def string_energy_vector(y, sr, n_fft=F_NFFT, hop=F_HOP, n_harm=5, cents_width=35):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    freqs_fft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    vec = np.zeros(16, dtype=np.float32)
    for s_idx in range(1, 17):
        f0 = HARP_STRINGS[s_idx]
        energy = 0.0
        for h in range(1, n_harm + 1):
            fh = f0 * h
            if fh >= freqs_fft[-1]:
                break
            band_lo = fh * (2 ** (-cents_width / 1200))
            band_hi = fh * (2 ** (cents_width / 1200))
            i0 = np.searchsorted(freqs_fft, band_lo)
            i1 = np.searchsorted(freqs_fft, band_hi)
            i1 = max(i1, i0 + 1)
            energy += S[i0:i1, :].mean()
        vec[s_idx - 1] = energy
    vec = np.log1p(vec)
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    return vec


def clip_to_mel_and_vec(y, sr=SAMPLE_RATE):
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]
    y = y / (np.max(np.abs(y)) + 1e-9)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    mel_db = mel_db[..., np.newaxis]
    vec = string_energy_vector(y, sr)
    return mel_db, vec


def yin_string_from_segment(seg, sr):
    f0 = librosa.yin(seg, fmin=90, fmax=800, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return None, None
    pitch = float(np.median(f0))
    string = int(np.argmin(np.abs(FREQS - pitch))) + 1
    return string, pitch


def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_time(t):
    """ASS time: H:MM:SS.cc (centiseconds)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = min(99, int(round((t - int(t)) * 100)))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# Fixed subtitle position (top-right). PlayRes 1920x1080; \pos(x,y) \an9 = anchor at (x,y).
ASS_PLAY_RES_X, ASS_PLAY_RES_Y = 1920, 1080
ASS_POS_X = ASS_PLAY_RES_X - 32  # 32px from right
ASS_POS_Y = 32  # 32px from top


def run_pipeline(model_path: str, video_path: str, output_dir: str, use_yin_fallback: bool = True):
    """
    Run harp string detection on a video.
    use_yin_fallback=False: default (model only, fixed threshold 0.25).
    use_yin_fallback=True: hybrid (per-string thresholds + YIN fallback).
    Returns (csv_path, labeled_video_path, dataframe).
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "audio_16k.wav")

    subprocess.run([
        FFMPEG_CMD, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        wav_path
    ], check=True, capture_output=True)

    model = tf.keras.models.load_model(model_path)

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)

    accepted = []
    last_t = -1e9
    for t in onsets:
        if (t - last_t) >= HOLD_SEC:
            accepted.append(float(t))
            last_t = float(t)
    onsets = np.array(accepted, dtype=float)

    thr = THR_ARRAY if use_yin_fallback else np.full(NUM_STRINGS, THR_DEFAULT, dtype=np.float32)

    rows = []
    for t in onsets:
        start = int((t + OFFSET_SEC) * sr)
        start = max(0, start)
        clip = y[start : start + CLIP_SAMPLES]
        mel, vec = clip_to_mel_and_vec(clip, sr)
        probs = model.predict([mel[None, ...], vec[None, ...]], verbose=0)[0]
        top1 = int(np.argmax(probs)) + 1
        top1_prob = float(np.max(probs))
        pred = (probs >= thr).astype(int)
        active = np.where(pred == 1)[0] + 1
        # Include top-1 if not already in active, and top-2 when above display threshold (so "4,9" can show)
        if top1 not in active:
            active = np.append(active, top1)
        sorted_idx = np.argsort(probs)[::-1]
        if len(sorted_idx) > 1:
            top2 = int(sorted_idx[1]) + 1
            top2_prob = float(probs[sorted_idx[1]])
            if top2 not in active and top2_prob >= TOP2_DISPLAY_THR:
                active = np.append(active, top2)
        active = np.unique(active).astype(int)
        used = "string"  # label: String (model)

        if use_yin_fallback and ((top1_prob < MODEL_CONF_MIN) or (len(active) == 0)):
            seg_end = int((t + YIN_WIN_SEC) * sr)
            seg = y[int(t * sr) : seg_end]
            s_yin, pitch = yin_string_from_segment(seg, sr)
            if s_yin is not None:
                active = np.array([s_yin], dtype=int)
                used = "string*"  # label: String* (YIN fallback)
            else:
                used = "none"

        row = {
            "time_sec": float(t),
            "predicted_strings": ",".join(map(str, active)) if len(active) else "",
            "top1": top1,
            "top1_prob": top1_prob,
            **{f"prob_S{i+1}": float(probs[i]) for i in range(NUM_STRINGS)},
            **{f"pred_S{i+1}": int(pred[i]) for i in range(NUM_STRINGS)},
        }
        if use_yin_fallback:
            row["used"] = used
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_name = "predictions_hybrid.csv" if use_yin_fallback else "predictions_default.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    # ASS overlay: fixed position (top-right) so label never moves. Full style header for libass.
    ass_path = os.path.join(output_dir, "overlay.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write(f"PlayResX: {ASS_PLAY_RES_X}\nPlayResY: {ASS_PLAY_RES_Y}\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # White text, black outline, Alignment 9 = top-right
        f.write("Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,9,0,32,32,1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for row in df.itertuples(index=False):
            t = float(row.time_sec)
            pred_str = (
                row.predicted_strings
                if isinstance(row.predicted_strings, str) and row.predicted_strings.strip()
                else "None"
            )
            used = getattr(row, "used", "string")
            used_label = "String*" if used == "string*" else "String" if used == "string" else used
            text = f"{used_label} → {pred_str}" if use_yin_fallback else f"t={t:.2f}s → {pred_str}"
            start_t = max(0.0, t - 0.02)
            end_t = start_t + FLASH_DURATION
            # Fixed pixel position: top-right anchor at (ASS_POS_X, ASS_POS_Y)
            line = f"Dialogue: 0,{ass_time(start_t)},{ass_time(end_t)},Default,,0,0,0,,{{\\pos({ASS_POS_X},{ASS_POS_Y})\\an9}}{text}\n"
            f.write(line)

    # Also write SRT for app.py fallback when using combined video
    srt_path = os.path.join(output_dir, "overlay.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            t = float(row.time_sec)
            pred_str = (
                row.predicted_strings
                if isinstance(row.predicted_strings, str) and row.predicted_strings.strip()
                else "None"
            )
            used = getattr(row, "used", "string")
            used_label = "String*" if used == "string*" else "String" if used == "string" else used
            text = f"{used_label} → {pred_str}" if use_yin_fallback else f"t={t:.2f}s → {pred_str}"
            start_t = max(0.0, t - 0.02)
            end_t = start_t + FLASH_DURATION
            f.write(f"{i}\n{srt_time(start_t)} --> {srt_time(end_t)}\n{text}\n\n")

    ass_abs = os.path.abspath(ass_path).replace("\\", "/")
    if os.name == "nt":
        ass_abs = ass_abs.replace(":", "\\:", 1)
    out_video = os.path.join(output_dir, "video_labeled.mp4")
    # Use subtitles filter (accepts ASS) so overlay is visible; ASS has fixed \pos in each line
    subprocess.run([
        FFMPEG_CMD, "-y", "-i", video_path,
        "-vf", f"subtitles='{ass_abs}'",
        "-c:a", "copy",
        out_video
    ], check=True, capture_output=True)

    return csv_path, out_video, df
