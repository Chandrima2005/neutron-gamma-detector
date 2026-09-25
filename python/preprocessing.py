"""
preprocessing.py
----------------
Signal preprocessing for the InDEx neutron/gamma acoustic discrimination
pipeline. This is the ONE place the waveform -> feature-vector logic lives
for the Python side; both train_model.py and predict.py import it, so
training and inference are guaranteed to use identical preprocessing.

Steps (matching the report's Section 2.2):
  1. Parse the raw LabVIEW (.lvm) waveform file
  2. Amplitude-normalize the signal
  3. Frame it (1024-sample Hanning window, 80% overlap)
  4. Compute short-time energy (STE) and locate the true nucleation pulse
     via an energy threshold
  5. Compute the STFT, normalize each frame's spectrum, and average over
     the pulse window -> a 512-dimensional feature vector
"""

import numpy as np

# ---- fixed pipeline constants (must match train_model.py / predict.py / the JS mirror) ----
N_FFT = 1024
OVERLAP = 0.8
HOP = int(N_FFT * (1 - OVERLAP))   # 204
THETA = 0.15                        # energy threshold for pulse localization


def read_lvm(path):
    """Parse a LabVIEW Measurement (.lvm) file.

    Returns
    -------
    signal : np.ndarray
        The waveform's amplitude values (the 'Trigger' column).
    delta_t : float
        The sampling interval in seconds (1 / sampling frequency).
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    header_idx = [i for i, l in enumerate(lines) if "End_of_Header" in l]
    if not header_idx:
        raise ValueError(f"{path}: not a recognized .lvm file (no 'End_of_Header' marker)")
    start = header_idx[-1] + 2  # skip the header marker line + the column-name row

    delta_x = None
    for l in lines:
        if l.startswith("Delta_X"):
            delta_x = float(l.split("\t")[1])
            break
    if delta_x is None:
        delta_x = 1e-6  # fall back to 1 MHz if not found

    values = []
    for l in lines[start:]:
        parts = l.strip().split("\t")
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue

    if len(values) < N_FFT:
        raise ValueError(f"{path}: only {len(values)} samples parsed, need at least {N_FFT}")

    return np.array(values), delta_x


def extract_features(signal, fs=1e6, n_fft=N_FFT, hop=HOP, theta=THETA):
    """Turn a raw waveform into a 512-D short-time spectral-energy feature vector.

    Returns a dict with the feature vector plus everything needed to plot
    the waveform / STE curve / spectrogram (used by predict.py's --plot
    option and by the dashboard).
    """
    xn = signal / np.max(np.abs(signal))                 # amplitude normalization

    n = len(xn)
    starts = list(range(0, n - n_fft, hop))
    frames = np.array([xn[s:s + n_fft] for s in starts])
    win = np.hanning(n_fft)
    xw = frames * win                                      # windowed frames

    E = np.mean(np.abs(xw) ** 2, axis=1)                    # short-time energy
    Ehat = E / np.max(E)                                    # normalized STE

    above = np.where(Ehat >= theta)[0]
    ks, ke = (int(above.min()), int(above.max())) if len(above) else (0, len(starts) - 1)

    X = np.fft.fft(xw, axis=1)[:, : n_fft // 2]             # STFT, keep 0..Nyquist
    P = (np.abs(X) ** 2) / n_fft                             # power spectrum
    Pmax = np.max(P, axis=1, keepdims=True)
    Pmax[Pmax == 0] = 1
    Phat = P / Pmax                                          # per-frame normalization

    feature = np.mean(Phat[ks:ke + 1], axis=0)               # temporal average -> 512-D

    frame_times = np.array(starts) / fs
    return {
        "feature": feature,           # (512,) — what the classifiers consume
        "raw": xn,                    # normalized waveform, for plotting
        "fs": fs,
        "Ehat": Ehat,                 # short-time energy curve
        "frame_times": frame_times,
        "ks": ks, "ke": ke,           # pulse-window frame indices
        "crop_start_t": float(frame_times[ks]),
        "crop_end_t": float(frame_times[ke] + n_fft / fs),
        "stft_P": Phat,                # full spectrogram (frames x 512), for plotting
    }


def featurize_file(path):
    """Convenience wrapper: read a .lvm file straight to a feature vector."""
    signal, dx = read_lvm(path)
    return extract_features(signal, fs=1.0 / dx)
