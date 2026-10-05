"""Mitigation: does training the probe on degraded examples recover accuracy?

Needs train embeddings under a few degradations first:
    python extract_embeddings.py --model phikon --train-conditions blur:2 blur:4 jpeg:30 jpeg:15 stain:0.3

Every probe sees the same number of training images. For an augmented probe each
training image is drawn at random from its clean version or one of its degraded versions.

Probes:
    clean       clean training images only (the baseline)
    aug_all     clean + every degraded train condition available
    aug_<kind>  clean + that kind only (shows whether the benefit transfers across kinds)

Each seed uses a different random 80% of the training images and a different random
mix of clean/degraded versions. The CSV reports the mean and standard deviation over seeds.

Outputs mitigation.csv and figures/mitigation_<model>_<kind>.png

Examples:
    python mitigation.py --models phikon phikon-v2 --seeds 5
    python mitigation.py --emb-dir embeddings/pcam --out-dir results_pcam --seeds 5
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import degradations


def load(path: Path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"]


def mixed_train_set(versions: list[np.ndarray], rng) -> np.ndarray:
    """One randomly chosen version (clean or degraded) per training image."""
    stack = np.stack(versions)                      # (n_versions, n_images, dim)
    choice = rng.integers(0, len(versions), stack.shape[1])
    return stack[choice, np.arange(stack.shape[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["phikon", "phikon-v2"])
    ap.add_argument("--emb-dir", type=Path, default=Path("embeddings"))
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic-regression probe")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="where the CSV and figures/ are written")
    ap.add_argument("--seeds", type=int, default=5, help="number of repeats")
    ap.add_argument("--train-fraction", type=float, default=0.8,
                    help="fraction of training images used per seed (1.0 = all, only the mixing varies)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = {}  # (model, probe, kind, level, seen) -> list of balanced accuracy, one per seed
    for name in args.models:
        d = args.emb_dir / name
        if not (d / "train_clean.npz").exists():
            print(f"{name}: no embeddings in {d}, skipping")
            continue
        X_clean, ytr = load(d / "train_clean.npz")
        degraded = {}  # condition -> train embeddings
        for cond in degradations.all_conditions()[1:]:
            f = d / f"train_{degradations.tag(cond)}.npz"
            if f.exists():
                degraded[cond] = load(f)[0]
        if not degraded:
            print(f"{name}: no degraded train embeddings found, run extract_embeddings.py with --train-conditions")
            continue
        print(f"{name}: degraded train conditions = {list(degraded)}")

        tests = {}
        for cond in degradations.all_conditions():
            f = d / f"test_{degradations.tag(cond)}.npz"
            if f.exists():
                tests[cond] = load(f)

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            n = len(ytr)
            keep = np.sort(rng.choice(n, int(round(args.train_fraction * n)), replace=False))
            train_sets = {"clean": X_clean, "aug_all": mixed_train_set([X_clean, *degraded.values()], rng)}
            for kind in degradations.LEVELS:
                same = [X for c, X in degraded.items() if c.startswith(kind + ":")]
                if same:
                    train_sets[f"aug_{kind}"] = mixed_train_set([X_clean, *same], rng)

            for probe_name, Xtr in train_sets.items():
                probe = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000))
                probe.fit(Xtr[keep], ytr[keep])
                for cond, (X, y) in tests.items():
                    kind, _, level = cond.partition(":")
                    seen = cond in degraded and (probe_name == "aug_all" or probe_name == f"aug_{kind}")
                    key = (name, probe_name, kind, level or "0", seen)
                    scores.setdefault(key, []).append(balanced_accuracy_score(y, probe.predict(X)))
            print(f"  seed {seed}: done")

    if not scores:
        return
    rows = [{
        "model": m, "probe": p, "kind": k, "level": lvl, "seen_in_training": seen,
        "balanced_accuracy": round(float(np.mean(v)), 4),
        "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4),
        "n_seeds": len(v),
    } for (m, p, k, lvl, seen), v in scores.items()]
    with open(args.out_dir / "mitigation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    xlabel = {"blur": "Gaussian blur sigma (px)", "jpeg": "JPEG quality", "stain": "Stain shift (alpha)"}
    for name in sorted({r["model"] for r in rows}):
        probes = list(dict.fromkeys(r["probe"] for r in rows if r["model"] == name))
        for kind, levels in degradations.LEVELS.items():
            ticks = ["clean"] + [str(v) for v in levels]
            xs = np.arange(len(ticks))
            fig, ax = plt.subplots(figsize=(6, 3.8))
            for p in probes:
                def pick(k, lvl, key):
                    hits = [r[key] for r in rows
                            if r["model"] == name and r["probe"] == p and r["kind"] == k and r["level"] == lvl]
                    return hits[0] if hits else np.nan
                ys = np.array([pick("clean", "0", "balanced_accuracy")] + [pick(kind, str(v), "balanced_accuracy") for v in levels])
                sd = np.array([pick("clean", "0", "std")] + [pick(kind, str(v), "std") for v in levels])
                ax.plot(xs, ys, marker="o", ms=4, label=p, linestyle="--" if p == "clean" else "-")
                ax.fill_between(xs, ys - sd, ys + sd, alpha=0.15)
            ax.set_xticks(xs, ticks)
            ax.set_xlabel(xlabel[kind])
            ax.set_ylabel("Balanced accuracy (mean ± sd over seeds)")
            ax.set_ylim(0, 1)
            ax.set_title(name)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / f"mitigation_{name}_{kind}.png", dpi=200)
            plt.close(fig)
    print(f"wrote {args.out_dir / 'mitigation.csv'} and {fig_dir}/mitigation_*.png")


if __name__ == "__main__":
    main()
