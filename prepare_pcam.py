"""Prepare a PatchCamelyon subset as PNG files plus manifests.

Download these four files from https://zenodo.org/records/2546921 into data/pcam/ first:
    camelyonpatch_level_2_split_valid_x.h5.gz   (806 MB)  -> used as the probe training pool
    camelyonpatch_level_2_split_valid_y.h5.gz
    camelyonpatch_level_2_split_test_x.h5.gz    (801 MB)  -> used as the test set
    camelyonpatch_level_2_split_test_y.h5.gz

The 6.4 GB official train split is not needed: the probe only uses a few thousand images.

Writes data/pcam/{train,test}/{normal,tumor}/*.png and pcam_{train,test}_manifest.csv.

Example:
    python prepare_pcam.py
"""
import argparse
import csv
import gzip
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

CLASSES = ["normal", "tumor"]


def open_h5(root: Path, split: str, which: str) -> h5py.File:
    h5 = root / f"camelyonpatch_level_2_split_{split}_{which}.h5"
    if not h5.exists():
        gz = h5.with_name(h5.name + ".gz")
        if not gz.exists():
            raise SystemExit(f"missing {gz} (download it from https://zenodo.org/records/2546921)")
        print(f"unzipping {gz.name} ...")
        with gzip.open(gz, "rb") as src, open(h5, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return h5py.File(h5, "r")


def export(root: Path, source_split: str, out_split: str, per_class: int, rng) -> list[tuple[str, str]]:
    y = np.asarray(open_h5(root, source_split, "y")["y"]).reshape(-1)
    x = open_h5(root, source_split, "x")["x"]
    rows = []
    for label, name in enumerate(CLASSES):
        idx = np.sort(rng.choice(np.flatnonzero(y == label), per_class, replace=False))
        out = root / out_split / name
        out.mkdir(parents=True, exist_ok=True)
        for i in idx:
            p = out / f"{source_split}_{i:06d}.png"
            if not p.exists():
                Image.fromarray(x[int(i)]).save(p)
            rows.append((p.as_posix(), name))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/pcam"))
    ap.add_argument("--train-per-class", type=int, default=4500)
    ap.add_argument("--test-per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for source, split, n in [("valid", "train", args.train_per_class), ("test", "test", args.test_per_class)]:
        rows = export(args.root, source, split, n, rng)
        out = Path(f"pcam_{split}_manifest.csv")
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)
        print(f"{out}: {len(rows)} images")


if __name__ == "__main__":
    main()
