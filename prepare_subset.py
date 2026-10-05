"""Build train/test manifests (CSV of path,label) from the unzipped Zenodo folders.

Example:
    python prepare_subset.py --train-dir data/NCT-CRC-HE-100K --test-dir data/CRC-VAL-HE-7K
"""
import argparse
import csv
import random
from pathlib import Path

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_images(root: Path) -> dict[str, list[Path]]:
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    return {c: sorted(p for p in (root / c).iterdir() if p.suffix.lower() in EXTS) for c in classes}


def write_manifest(rows: list[tuple[Path, str]], out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows((str(p), c) for p, c in rows)
    print(f"{out}: {len(rows)} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = []
    for c, paths in list_images(args.train_dir).items():
        train += [(p, c) for p in rng.sample(paths, min(args.per_class, len(paths)))]
    test = [(p, c) for c, paths in list_images(args.test_dir).items() for p in paths]

    write_manifest(train, Path("train_manifest.csv"))
    write_manifest(test, Path("test_manifest.csv"))


if __name__ == "__main__":
    main()
