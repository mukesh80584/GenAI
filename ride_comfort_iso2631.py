#!/usr/bin/env python3
"""
ISO 2631-1 Longitudinal (X-axis) Ride Comfort Analysis

This script reads time-series longitudinal acceleration data from CSV,
preprocesses it, applies ISO 2631-style weighting (Wd approximation), and
computes ride comfort metrics:
  - Weighted RMS acceleration (aw_rms)
  - Crest factor (CF)
  - Vibration Dose Value (VDV)
  - RMS jerk (j_rms)

Usage:
    python ride_comfort_iso2631.py --input data.csv --time-col Time --acc-col ax
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


@dataclass
class RideComfortMetrics:
    """Container for computed ride comfort metrics."""

    aw_rms: float
    crest_factor: float
    vdv: float
    j_rms: float



def read_data(file_path: str, time_col: str = "Time", acc_col: str = "ax") -> pd.DataFrame:
    """
    Read and clean input CSV data.

    Steps:
      1) Read CSV
      2) Keep required columns
      3) Remove NaN/non-finite values
      4) Sort by time and drop duplicate timestamps

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns [time_col, acc_col].
    """
    df = pd.read_csv(file_path)

    if time_col not in df.columns or acc_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{time_col}' and '{acc_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[time_col, acc_col]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df = df.sort_values(time_col).drop_duplicates(subset=time_col).reset_index(drop=True)

    if len(df) < 4:
        raise ValueError("Insufficient valid samples after cleaning. Need at least 4 samples.")

    return df



def ensure_uniform_sampling(time: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Ensure uniform sampling by linear interpolation when needed.

    Returns
    -------
    t_uniform : np.ndarray
    a_uniform : np.ndarray
    fs        : float
        Estimated sampling frequency (Hz).
    """
    dt = np.diff(time)
    dt_med = np.median(dt)

    if dt_med <= 0:
        raise ValueError("Non-positive median time step detected.")

    fs = 1.0 / dt_med

    # Uniformity check: if jitter > 1% of median, resample
    if np.max(np.abs(dt - dt_med)) > 0.01 * dt_med:
        t_uniform = np.arange(time[0], time[-1] + 0.5 * dt_med, dt_med)
        a_uniform = np.interp(t_uniform, time, acc)
    else:
        t_uniform = time
        a_uniform = acc

    return t_uniform, a_uniform, fs



def filter_signal(acc: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 80.0, order: int = 4) -> np.ndarray:
    """
    Apply Wd-like weighting using a Butterworth band-pass approximation.

    ISO 2631 Wd is a specific analog weighting. If exact implementation is not
    available, this approximation uses band-pass 0.5-80 Hz for longitudinal axis.
    Zero-phase filtering is used to avoid phase distortion.
    """
    nyq = 0.5 * fs

    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)

    if low >= high:
        raise ValueError(
            f"Invalid band edges for fs={fs:.3f} Hz. "
            "Increase sampling frequency or adjust cutoffs."
        )

    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, acc)



def compute_metrics(time: np.ndarray, aw: np.ndarray, raw_acc: np.ndarray) -> RideComfortMetrics:
    """Compute ISO 2631-related ride comfort parameters."""
    duration = time[-1] - time[0]
    if duration <= 0:
        raise ValueError("Time duration must be positive.")

    # Weighted RMS acceleration
    aw_rms = np.sqrt(np.trapz(aw**2, time) / duration)

    # Crest Factor
    crest_factor = np.max(np.abs(aw)) / aw_rms if aw_rms > 0 else np.nan

    # Vibration Dose Value
    vdv = np.trapz(np.abs(aw) ** 4, time) ** 0.25

    # Jerk from raw acceleration
    jerk = np.gradient(raw_acc, time)
    j_rms = np.sqrt(np.mean(jerk**2))

    return RideComfortMetrics(aw_rms=aw_rms, crest_factor=crest_factor, vdv=vdv, j_rms=j_rms)



def classify_comfort(aw_rms: float) -> str:
    """Classify comfort level based on weighted RMS thresholds."""
    if aw_rms < 0.315:
        return "Comfortable"
    if aw_rms <= 0.63:
        return "Slight discomfort"
    return "Uncomfortable"



def plot_results(time: np.ndarray, acc: np.ndarray, aw: np.ndarray, jerk: np.ndarray, spike_sigma: float = 3.0) -> None:
    """Plot acceleration, weighted acceleration, and jerk with spike highlights."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, acc, color="tab:blue", lw=1.1)
    axes[0].set_ylabel("a_x (m/s²)")
    axes[0].set_title("Longitudinal Acceleration")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, aw, color="tab:green", lw=1.1)
    axes[1].set_ylabel("a_w (m/s²)")
    axes[1].set_title("Weighted Acceleration (Wd Approximation)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, jerk, color="tab:red", lw=1.0, label="Jerk")

    # Spike detection using robust threshold
    j_abs = np.abs(jerk)
    thr = np.mean(j_abs) + spike_sigma * np.std(j_abs)
    spikes = np.where(j_abs >= thr)[0]
    if len(spikes) > 0:
        axes[2].scatter(time[spikes], jerk[spikes], color="black", s=18, label="Jerk spikes")

    axes[2].axhline(thr, color="gray", ls="--", lw=1.0, alpha=0.8)
    axes[2].axhline(-thr, color="gray", ls="--", lw=1.0, alpha=0.8)
    axes[2].set_ylabel("j (m/s³)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Jerk with Spike Highlights")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()



def main() -> None:
    parser = argparse.ArgumentParser(description="ISO 2631 Longitudinal Ride Comfort Analyzer")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--time-col", default="Time", help="Time column name (default: Time)")
    parser.add_argument("--acc-col", default="ax", help="Acceleration column name (default: ax)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    df = read_data(args.input, args.time_col, args.acc_col)

    time = df[args.time_col].to_numpy(dtype=float)
    acc = df[args.acc_col].to_numpy(dtype=float)

    t_u, a_u, fs = ensure_uniform_sampling(time, acc)
    aw = filter_signal(a_u, fs=fs)

    jerk = np.gradient(a_u, t_u)
    metrics = compute_metrics(t_u, aw, a_u)

    comfort = classify_comfort(metrics.aw_rms)

    print("\n=== ISO 2631 Longitudinal Ride Comfort Results ===")
    print(f"Estimated sampling frequency : {fs:.3f} Hz")
    print(f"Weighted RMS, aw_rms         : {metrics.aw_rms:.6f} m/s²")
    print(f"Crest Factor, CF             : {metrics.crest_factor:.6f}")
    print(f"Vibration Dose Value, VDV    : {metrics.vdv:.6f} m/s^1.75")
    print(f"RMS Jerk, j_rms              : {metrics.j_rms:.6f} m/s³")
    print(f"Comfort classification        : {comfort}")

    if not args.no_plot:
        plot_results(t_u, a_u, aw, jerk)


if __name__ == "__main__":
    main()
