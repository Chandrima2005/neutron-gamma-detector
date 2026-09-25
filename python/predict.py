"""
predict.py
----------
Applies a trained model to a NEW .lvm waveform: signal preprocessing
(preprocessing.py) -> 512-D feature vector -> linear model -> class.

Both models are linear (SGDClassifier), so "using the model" is just:
    score = w . feature + b
    Logistic Regression : P(neutron) = sigmoid(score)   -> Neutron if >= 0.5
    Linear SVM           : decision  = score             -> Neutron if >= 0

Usage:
    python predict.py --weights model_weights.json --input scope-1_026.lvm
    python predict.py --weights model_weights.json --input scope-1_026.lvm --plot out.png
"""

import json
import argparse
import numpy as np

from preprocessing import featurize_file


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(feature, weights):
    """feature: (512,) np.array. weights: dict loaded from model_weights.json."""
    lr_w, lr_b = np.array(weights["lr"]["w"]), weights["lr"]["b"]
    svm_w, svm_b = np.array(weights["svm"]["w"]), weights["svm"]["b"]

    lr_score = float(np.dot(lr_w, feature) + lr_b)
    lr_proba = sigmoid(lr_score)
    svm_score = float(np.dot(svm_w, feature) + svm_b)

    return {
        "lr_pred": "Neutron" if lr_proba >= 0.5 else "Gamma",
        "lr_proba_neutron": lr_proba,
        "svm_pred": "Neutron" if svm_score >= 0 else "Gamma",
        "svm_score": svm_score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights.json")
    ap.add_argument("--input", required=True, help="path to a .lvm waveform file")
    ap.add_argument("--plot", default=None, help="optional path to save a waveform/spectrogram figure (requires matplotlib)")
    args = ap.parse_args()

    with open(args.weights) as f:
        weights = json.load(f)

    r = featurize_file(args.input)
    result = predict(r["feature"], weights)

    print(f"File:              {args.input}")
    print(f"Pulse window:      {r['crop_start_t']*1000:.3f}-{r['crop_end_t']*1000:.3f} ms")
    print(f"Logistic Regression -> {result['lr_pred']}  (P(neutron) = {result['lr_proba_neutron']:.3f})")
    print(f"Linear SVM           -> {result['svm_pred']}  (decision score = {result['svm_score']:.3f})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
        t_ms = np.arange(len(r["raw"])) / r["fs"] * 1000
        axes[0].plot(t_ms, r["raw"], lw=0.6)
        axes[0].axvspan(r["crop_start_t"]*1000, r["crop_end_t"]*1000, color="orange", alpha=0.2)
        axes[0].set_title("Waveform"); axes[0].set_xlabel("ms")

        axes[1].plot(r["frame_times"]*1000, r["Ehat"], lw=1)
        axes[1].axhline(0.15, color="red", ls="--", lw=0.8)
        axes[1].set_title("Short-time energy"); axes[1].set_xlabel("ms")

        axes[2].imshow(r["stft_P"].T, aspect="auto", origin="lower", cmap="inferno",
                        extent=[0, len(r["raw"])/r["fs"]*1000, 0, r["fs"]/2/1000])
        axes[2].set_title("STFT spectrogram"); axes[2].set_xlabel("ms"); axes[2].set_ylabel("kHz")

        fig.suptitle(f"{args.input} — LR: {result['lr_pred']} | SVM: {result['svm_pred']}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"Saved figure to {args.plot}")


if __name__ == "__main__":
    main()
