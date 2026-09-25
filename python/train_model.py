"""
train_model.py
---------------
Trains the two classifiers (Logistic Regression, Linear SVM) on a folder
of labelled .lvm waveforms, using preprocessing.py for feature extraction.

Usage:
    python train_model.py --data-dir "/path/to/Data_100 events" --epochs 30000

Expected folder layout:
    <data-dir>/Neutron/*.lvm
    <data-dir>/Gamma/*.lvm

Outputs (written next to this script unless --out-dir is given):
    model_weights.json   -- trained weights, for predict.py / re-analysis
    model_weights.js      -- the SAME weights as a JS file the web UI loads
    training_report.json  -- accuracy / precision / recall / F1 / confusion matrices / loss curves
"""

import os
import glob
import json
import argparse
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from preprocessing import featurize_file

CLASSES = {"Neutron": 1, "Gamma": 0}  # Class A = Neutron, Class B = Gamma


def load_dataset(data_dir):
    X, y, meta = [], [], []
    for cname, clabel in CLASSES.items():
        files = sorted(glob.glob(os.path.join(data_dir, cname, "*.lvm")))
        if not files:
            raise FileNotFoundError(f"No .lvm files found under {os.path.join(data_dir, cname)}")
        for fp in files:
            r = featurize_file(fp)
            X.append(r["feature"])
            y.append(clabel)
            meta.append({"file": os.path.basename(fp), "class": cname})
    return np.array(X), np.array(y), meta


def train_sgd(X_train, y_train, loss, eta0, power_t, epochs, record_every=150):
    clf = SGDClassifier(
        loss=loss, learning_rate="invscaling", eta0=eta0, power_t=power_t,
        max_iter=1, tol=None, warm_start=True, random_state=42,
    )
    classes = np.unique(y_train)
    losses = []
    for epoch in range(epochs):
        clf.partial_fit(X_train, y_train, classes=classes)
        if epoch % record_every == 0 or epoch == epochs - 1:
            if loss == "log_loss":
                p = np.clip(clf.predict_proba(X_train)[:, 1], 1e-12, 1 - 1e-12)
                L = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
            else:
                scores = clf.decision_function(X_train)
                yy = np.where(y_train == 1, 1, -1)
                L = np.mean(np.maximum(0, 1 - yy * scores))
            losses.append((epoch, float(L)))
    return clf, losses


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[1, 0], zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    return {
        "accuracy": float(acc),
        "precision": {"Neutron": float(prec[0]), "Gamma": float(prec[1])},
        "recall": {"Neutron": float(rec[0]), "Gamma": float(rec[1])},
        "f1": {"Neutron": float(f1[0]), "Gamma": float(f1[1])},
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=30000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading and featurizing waveforms from {args.data_dir} ...")
    X, y, meta = load_dataset(args.data_dir)
    print(f"  -> {len(X)} events, feature dim = {X.shape[1]}")

    idx_all = np.arange(len(X))
    idx_train, idx_temp, y_train, y_temp = train_test_split(idx_all, y, test_size=0.3, stratify=y, random_state=42)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
    X_train, X_test = X[idx_train], X[idx_test]
    print(f"  -> train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    print("Training Logistic Regression (SGD, log-loss) ...")
    lr_clf, lr_losses = train_sgd(X_train, y_train, "log_loss", eta0=1e-4, power_t=0.5, epochs=args.epochs)
    lr_eval = evaluate(lr_clf, X_test, y_test)
    print(f"  LR  accuracy = {lr_eval['accuracy']*100:.2f}%  |  confusion = {lr_eval['confusion_matrix']}")

    print("Training Linear SVM (SGD, hinge-loss) ...")
    svm_clf, svm_losses = train_sgd(X_train, y_train, "hinge", eta0=1e-3, power_t=0.5, epochs=args.epochs)
    svm_eval = evaluate(svm_clf, X_test, y_test)
    print(f"  SVM accuracy = {svm_eval['accuracy']*100:.2f}%  |  confusion = {svm_eval['confusion_matrix']}")

    # ---- export weights (both models are linear: prediction = w . feature + b) ----
    weights = {
        "lr":  {"w": lr_clf.coef_[0].tolist(),  "b": float(lr_clf.intercept_[0])},
        "svm": {"w": svm_clf.coef_[0].tolist(), "b": float(svm_clf.intercept_[0])},
    }
    weights_json_path = os.path.join(args.out_dir, "model_weights.json")
    with open(weights_json_path, "w") as f:
        json.dump(weights, f)
    print(f"Saved {weights_json_path}")

    # same weights, wrapped as a plain JS file the browser UI can <script src=...> directly
    weights_js_path = os.path.join(args.out_dir, "model_weights.js")
    with open(weights_js_path, "w") as f:
        f.write("// Auto-generated by train_model.py — trained Logistic Regression + Linear SVM weights\n")
        f.write("const MODEL_WEIGHTS = " + json.dumps(weights) + ";\n")
    print(f"Saved {weights_js_path}")

    report = {
        "n_events": len(X), "feature_dim": int(X.shape[1]), "epochs": args.epochs,
        "split": {"n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test)},
        "lr": {"eval": lr_eval, "loss_curve": lr_losses},
        "svm": {"eval": svm_eval, "loss_curve": svm_losses},
    }
    report_path = os.path.join(args.out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
