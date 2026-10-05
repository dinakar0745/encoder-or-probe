"""Drift and per-class failure analysis with repeated seeds.

For every model and degraded test condition, using a probe trained on CLEAN images only:
  - embedding drift from the clean version of the same image (cosine distance)
  - how often the prediction flips, and how well per-image drift ranks the flipped images (AUROC)
  - recall of every class

Each seed trains the probe on a different random 80% of the training images.
Needs no new extraction.

Outputs (in --out-dir):
    drift.csv       mean drift, balanced accuracy, accuracy drop, flip rate, AUROC (mean and sd over seeds)
    per_class.csv   recall per class (mean and sd over seeds)

Examples:
    python failure_analysis.py --models phikon phikon-v2 dinov2-l kaiko-b16 --out-dir results_nct_c1
    python failure_analysis.py --models phikon phikon-v2 dinov2-l kaiko-b16 --C 0.001 --emb-dir embeddings/pcam --out-dir results_pcam_c001
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import degradations


def load(path: Path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - (a * b).sum(axis=1)


def ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return round(float(v.mean()), 4), round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["phikon", "phikon-v2"])
    ap.add_argument("--emb-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic-regression probe")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-fraction", type=float, default=0.8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drift_rows, class_rows = [], []
    for name in args.models:
        d = args.emb_dir / name
        if not (d / "train_clean.npz").exists():
            print(f"{name}: no embeddings in {d}, skipping")
            continue
        Xtr, ytr = load(d / "train_clean.npz")
        tests = {}
        for cond in degradations.all_conditions():
            f = d / f"test_{degradations.tag(cond)}.npz"
            if f.exists():
                tests[cond] = load(f)
        X0, y = tests["clean"]
        classes = sorted(set(ytr))
        drift = {c: cosine_distance(X0, X) for c, (X, _) in tests.items() if c != "clean"}

        acc, flip, auroc, recall = {}, {}, {}, {}
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(ytr), int(round(args.train_fraction * len(ytr))), replace=False))
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000))
            probe.fit(Xtr[keep], ytr[keep])
            pred0 = probe.predict(X0)
            for cond, (X, _) in tests.items():
                pred = pred0 if cond == "clean" else probe.predict(X)
                rec = [float((pred[y == c] == c).mean()) for c in classes]
                acc.setdefault(cond, []).append(float(np.mean(rec)))
                for c, r in zip(classes, rec):
                    recall.setdefault((cond, c), []).append(r)
                if cond != "clean":
                    flipped = pred != pred0
                    flip.setdefault(cond, []).append(float(flipped.mean()))
                    ok = 0 < flipped.sum() < len(flipped)
                    auroc.setdefault(cond, []).append(roc_auc_score(flipped, drift[cond]) if ok else float("nan"))
            print(f"{name}: seed {seed} done")

        clean_acc = np.array(acc["clean"])
        for cond in tests:
            kind, _, level = cond.partition(":")
            for c in classes:
                m, s = ms(recall[(cond, c)])
                class_rows.append({"model": name, "kind": kind, "level": level or "0", "class": c,
                                   "recall": m, "std": s, "n_seeds": args.seeds})
            if cond == "clean":
                continue
            a_m, a_s = ms(acc[cond])
            d_m, d_s = ms(clean_acc - np.array(acc[cond]))
            f_m, f_s = ms(flip[cond])
            u_m, u_s = ms(auroc[cond])
            drift_rows.append({
                "model": name, "kind": kind, "level": level,
                "mean_drift": round(float(drift[cond].mean()), 4),
                "median_drift": round(float(np.median(drift[cond])), 4),
                "balanced_accuracy": a_m, "balanced_accuracy_std": a_s,
                "accuracy_drop": d_m, "accuracy_drop_std": d_s,
                "flip_rate": f_m, "flip_rate_std": f_s,
                "auroc_drift_flags_flip": u_m, "auroc_std": u_s,
                "n_seeds": args.seeds,
            })

    for fname, rows in [("drift.csv", drift_rows), ("per_class.csv", class_rows)]:
        if rows:
            with open(args.out_dir / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {args.out_dir / fname}")


if __name__ == "__main__":
    main()
