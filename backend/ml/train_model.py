"""
BITalino ECG+EDA Stress Prototype (Isolation Forest)
- Expects OpenSignals .txt files with a JSON header and tab-separated data.
- Forces: ECG = A1, EDA = A2
- Assumes: sampling rate = 1000 Hz (fs)

Folder structure (relative to this script):
  CALM/
    baseline_andi_ecg_eda.txt
    baseline_orion_ecg_eda.txt
  STRESS/
    stress_andi_ecg_eda.txt
    stress_orion_ecg_eda.txt

Run:
  python train_model.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt
from sklearn.ensemble import IsolationForest


# -----------------------------
# 1) Read OpenSignals .txt
# -----------------------------
def read_opensignals_txt(path: str):
    """
    Reads an OpenSignals Text File Format (.txt).
    Returns: (df, meta)
      - df: DataFrame with columns like nSeq, I1, I2, O1, O2, A1..A6
      - meta: device metadata dict (contains sampling rate, columns, labels, etc.)
    """
    path = Path(path)

    header_json_line = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("# {"):
                header_json_line = line[2:].strip()  # remove "# "
            if line.strip() == "# EndOfHeader":
                break

    if header_json_line is None:
        raise ValueError(f"No OpenSignals JSON header found in: {path}")

    meta_all = json.loads(header_json_line)
    device_key = next(iter(meta_all.keys()))
    meta = meta_all[device_key]

    cols = meta.get("column", None)
    if cols is None:
        raise ValueError(f"Header missing 'column' field in: {path}")

    # Read numeric data (skip header lines starting with '#')
    df = pd.read_csv(path, sep=r"\t+", engine="python", comment="#", header=None)
    df = df.dropna(axis=1, how="all")  # drop empty trailing column if present

    # Align columns safely
    if df.shape[1] != len(cols):
        m = min(df.shape[1], len(cols))
        df = df.iloc[:, :m]
        cols = cols[:m]

    df.columns = cols
    return df, meta


# -----------------------------
# 2) Filtering helpers
# -----------------------------
def butter_filter(x, fs, low=None, high=None, order=4):
    nyq = 0.5 * fs

    if low is not None and high is not None:
        btype = "bandpass"
        Wn = [low / nyq, high / nyq]
    elif low is not None:
        btype = "highpass"
        Wn = low / nyq
    elif high is not None:
        btype = "lowpass"
        Wn = high / nyq
    else:
        return x

    b, a = butter(order, Wn, btype=btype)
    return filtfilt(b, a, x)


def preprocess_ecg(ecg, fs):
    """
    Basic ECG cleaning:
    - remove mean
    - bandpass 0.5–40 Hz
    """
    ecg = np.asarray(ecg, dtype=float)
    ecg = ecg - np.nanmean(ecg)
    ecg = butter_filter(ecg, fs, low=0.5, high=40, order=4)
    return ecg


def preprocess_eda(eda, fs):
    """
    Basic EDA cleaning (slow-changing signal):
    - remove mean
    - lowpass <= 5 Hz
    - median filter to reduce spikes
    """
    eda = np.asarray(eda, dtype=float)
    eda = eda - np.nanmean(eda)
    eda = butter_filter(eda, fs, high=5, order=4)
    eda = medfilt(eda, kernel_size=5)
    return eda


# -----------------------------
# 3) Windowing + features
# -----------------------------
def window_signal(x, fs, win_sec=5.0, step_sec=2.5):
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    if win <= 0 or step <= 0:
        raise ValueError("Bad window or step size.")

    windows = []
    for start in range(0, len(x) - win + 1, step):
        windows.append(x[start : start + win])
    return windows


def features_from_window(w):
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "energy": 0.0}

    mean = float(np.mean(w))
    std = float(np.std(w))
    var = float(np.var(w))
    energy = float(np.mean(w ** 2))  # average energy
    return {"mean": mean, "std": std, "var": var, "energy": energy}


def build_feature_table(ecg, eda, fs, win_sec=5.0, step_sec=2.5):
    ecg_wins = window_signal(ecg, fs, win_sec, step_sec)
    eda_wins = window_signal(eda, fs, win_sec, step_sec)

    n = min(len(ecg_wins), len(eda_wins))
    rows = []

    for i in range(n):
        f_ecg = features_from_window(ecg_wins[i])
        f_eda = features_from_window(eda_wins[i])

        row = {f"ecg_{k}": v for k, v in f_ecg.items()}
        row.update({f"eda_{k}": v for k, v in f_eda.items()})
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# 4) File -> features (FORCED A1/A2, fs=1000)
# -----------------------------
FS = 1000
ECG_COL = "A1"
EDA_COL = "A2"


def features_from_file(path: str):
    df, _meta = read_opensignals_txt(path)

    # Hard checks (prevents training on wrong columns)
    if ECG_COL not in df.columns:
        raise ValueError(f"[{path}] Missing ECG column '{ECG_COL}'. Found columns: {list(df.columns)}")
    if EDA_COL not in df.columns:
        raise ValueError(f"[{path}] Missing EDA column '{EDA_COL}'. Found columns: {list(df.columns)}")

    ecg = preprocess_ecg(df[ECG_COL].to_numpy(), FS)
    eda = preprocess_eda(df[EDA_COL].to_numpy(), FS)

    feats = build_feature_table(ecg, eda, FS, win_sec=5.0, step_sec=2.5)
    if feats.empty:
        raise ValueError(f"[{path}] No windows produced. Is the recording too short?")
    return feats


# -----------------------------
# 5) Isolation Forest: train + score
# -----------------------------
def train_isoforest(baseline_feature_tables):
    X = pd.concat(baseline_feature_tables, ignore_index=True).to_numpy()

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )
    model.fit(X)
    return model


def score_anomalies(model, feats: pd.DataFrame):
    """
    IsolationForest:
      decision_function: higher = more normal
    We'll output anomaly_score where higher = more "different" (more stress-like / unusual)
    """
    X = feats.to_numpy()
    normality = model.decision_function(X)
    anomaly_score = -normality
    preds = model.predict(X)  # 1 normal, -1 anomaly
    return anomaly_score, preds


# -----------------------------
# 6) Main runner
# -----------------------------
def main():
    calm_files = sorted(glob.glob("CALM/*.txt"))
    stress_files = sorted(glob.glob("STRESS/*.txt"))

    if not calm_files:
        raise FileNotFoundError("No baseline files found in CALM/*.txt")
    if not stress_files:
        print("Note: No stress files found in STRESS/*.txt (training will still work).")

    print("Baseline (CALM) files:")
    for f in calm_files:
        print("  -", f)

    print("\nStress files:")
    for f in stress_files:
        print("  -", f)

    # --- Build baseline features ---
    baseline_tables = []
    total_windows = 0
    for f in calm_files:
        feats = features_from_file(f)
        baseline_tables.append(feats)
        total_windows += len(feats)
        print(f"\n[CALM] {f}: windows={len(feats)}")

    # --- Train model on baseline only ---
    model = train_isoforest(baseline_tables)
    print(f"\n✅ Trained Isolation Forest on baseline windows: {total_windows}")

    # --- Score each stress file (and also score calm files if you want) ---
    def report(file_path, tag):
        feats = features_from_file(file_path)
        scores, preds = score_anomalies(model, feats)

        n_anom = int(np.sum(preds == -1))
        ratio = (n_anom / len(preds)) if len(preds) else 0.0

        print(f"\n[{tag}] {file_path}")
        print(f"  windows: {len(preds)}")
        print(f"  anomalies flagged: {n_anom} ({ratio*100:.1f}%)")
        print(f"  anomaly_score mean: {float(np.mean(scores)):.4f}")
        print(f"  anomaly_score max : {float(np.max(scores)):.4f}")

        # Show first 10 windows for quick sanity check
        print("  first 10 windows:")
        for i in range(min(10, len(scores))):
            label = "ANOM" if preds[i] == -1 else "OK"
            print(f"    - w{i:03d}: score={scores[i]:.4f}  {label}")

    # Score stress files
    for f in stress_files:
        report(f, "STRESS")

    # Optional: also score calm files to see baseline behavior (should be mostly OK)
    print("\n(Optional) Scoring baseline files too (should be mostly OK):")
    for f in calm_files:
        report(f, "CALM")


if __name__ == "__main__":
    main()