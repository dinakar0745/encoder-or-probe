"""Sanity check for the linear probe on clean data: is the clean accuracy limited by
probe regularisation or by the size of the training subset?

1. Sweeps the regularisation strength C, choosing the best on a held-out 20% of the
   TRAIN set (never the test set), and reports test accuracy for every C.
2. At the chosen C, reports a learning curve over training-set size.

Needs no new extraction. Writes probe_sanity.csv.

Examples:
    python probe_sanity.py
    python probe_sanity.py --emb-dir embeddings/pcam --out-dir results_pcam
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def load(path: Path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"]


def fit_score(Xtr, ytr, Xte, yte, C):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    probe.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, probe.predict(Xte)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["phikon", "phikon-v2"])
    ap.add_argument("--emb-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.models:
        d = args.emb_dir / name
        if not (d / "train_clean.npz").exists():
            print(f"{name}: no embeddings in {d}, skipping")
            continue
        Xtr, ytr = load(d / "train_clean.npz")
        Xte, yte = load(d / "test_clean.npz")
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=0)

        print(f"\n{name}: regularisation sweep ({len(ytr)} train, {len(yte)} test)")
        best_C, best_val = None, -1.0
        for C in CS:
            val = fit_score(Xa, ya, Xv, yv, C)
            test = fit_score(Xtr, ytr, Xte, yte, C)
            rows.append({"model": name, "check": "C_sweep", "C": C, "train_size": len(ytr),
                         "val_balanced_accuracy": round(val, 4), "test_balanced_accuracy": round(test, 4)})
            print(f"  C={C:<7} held-out train {val:.4f}   test {test:.4f}")
            if val > best_val:
                best_C, best_val = C, val
        print(f"  chosen on held-out train data: C={best_C}  (the pipeline default is C=1.0)")

        print(f"{name}: learning curve at C={best_C}")
        for frac in FRACTIONS:
            idx = np.arange(len(ytr)) if frac == 1.0 else train_test_split(
                np.arange(len(ytr)), train_size=frac, stratify=ytr, random_state=0)[0]
            test = fit_score(Xtr[idx], ytr[idx], Xte, yte, best_C)
            rows.append({"model": name, "check": "learning_curve", "C": best_C, "train_size": len(idx),
                         "val_balanced_accuracy": "", "test_balanced_accuracy": round(test, 4)})
            print(f"  {len(idx):>5} train images: test {test:.4f}")

    if rows:
        out = args.out_dir / "probe_sanity.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
