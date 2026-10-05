"""Extract frozen CLS embeddings and cache them to embeddings/<model>/<split>_<condition>.npz.

Finished conditions are skipped, so the script can be stopped and restarted freely.

Examples:
    python extract_embeddings.py --model phikon --limit 100      # smoke test
    python extract_embeddings.py --model phikon                  # full sweep
    python extract_embeddings.py --model phikon-v2
    python extract_embeddings.py --model phikon --train-conditions blur:2 blur:4 jpeg:30 jpeg:15 stain:0.3
    python extract_embeddings.py --model phikon --dataset pcam --train-conditions blur:2 blur:4 jpeg:30 jpeg:15 stain:0.3

Models marked "timm" below load through the timm library, which needs torchvision. Run those
from a separate virtual environment (see TIMM_MODELS) so the main environment stays unchanged.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import degradations

MODELS = {
    "phikon": "owkin/phikon",            # ViT-B/16, iBOT, pathology
    "phikon-v2": "owkin/phikon-v2",      # ViT-L/16, DINOv2, pathology
    "dinov2-l": "facebook/dinov2-large",  # ViT-L/14, DINOv2, natural images (baseline matching phikon-v2)
    "dinov2-b": "facebook/dinov2-base",   # ViT-B/14, DINOv2, natural images
}

# Loaded with timm (needs torchvision): use a separate environment, e.g.
#   python -m venv .venv-timm  &&  pip install torch torchvision timm numpy pillow scikit-image
TIMM_MODELS = {
    # ViT-B/16, DINO, pathology (Kaiko.ai, TCGA); freely downloadable, non-commercial licence
    "kaiko-b16": "hf_hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
}


def load_model(name: str, device: str):
    """Returns (model, kind, preprocess). kind is "hf" or "timm"."""
    if name in TIMM_MODELS:
        import timm
        model = timm.create_model(TIMM_MODELS[name], pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        cfg["crop_pct"] = 1.0  # resize to 224 then centre-crop 224, as on the model card
        print(f"timm preprocessing: {cfg}")
        return model, "timm", timm.data.create_transform(**cfg, is_training=False)
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODELS[name])
    model = AutoModel.from_pretrained(MODELS[name]).eval().to(device)
    return model, "hf", processor


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_manifest(path: str, limit: int | None) -> tuple[list[str], list[str]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if limit:
        step = max(1, len(rows) // limit)  # spread across classes
        rows = rows[::step][:limit]
    return [r["path"] for r in rows], [r["label"] for r in rows]


class PatchDataset(Dataset):
    def __init__(self, paths, condition, kind, preprocess):
        self.paths, self.condition, self.kind, self.preprocess = paths, condition, kind, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = degradations.apply(img, self.condition)
        if self.kind == "timm":
            return self.preprocess(img)
        return self.preprocess(images=img, return_tensors="pt")["pixel_values"][0]


@torch.inference_mode()
def embed(model, kind, loader, device) -> np.ndarray:
    out = []
    for batch in loader:
        batch = batch.to(device)
        # timm returns the pooled CLS feature directly; transformers returns all tokens.
        feats = model(batch) if kind == "timm" else model(pixel_values=batch).last_hidden_state[:, 0]
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[*MODELS, *TIMM_MODELS], required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="use only N images per split (smoke test)")
    ap.add_argument("--dataset", default=None,
                    help="dataset tag, e.g. pcam: reads <tag>_train_manifest.csv / <tag>_test_manifest.csv "
                         "and writes to embeddings/<tag>/<model>/ (default: the original NCT-CRC-HE layout)")
    ap.add_argument("--train-conditions", nargs="*", default=[],
                    help='also embed the train split under these conditions, e.g. blur:2 jpeg:15 (for mitigation.py)')
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model, kind, preprocess = load_model(args.model, device)

    prefix = f"{args.dataset}_" if args.dataset else ""
    emb_root = Path("embeddings") / args.dataset if args.dataset else Path("embeddings")
    out_dir = emb_root / (args.model + ("_smoke" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train split is embedded clean only; the test split is embedded under every condition.
    jobs = [("train", "clean")] + [("train", c) for c in args.train_conditions]
    jobs += [("test", c) for c in degradations.all_conditions()]
    for split, condition in jobs:
        out = out_dir / f"{split}_{degradations.tag(condition)}.npz"
        if out.exists():
            print(f"skip {out}")
            continue
        paths, labels = read_manifest(f"{prefix}{split}_manifest.csv", args.limit)
        loader = DataLoader(
            PatchDataset(paths, condition, kind, preprocess),
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        t0 = time.time()
        X = embed(model, kind, loader, device)
        np.savez(out, X=X, y=np.array(labels))
        dt = time.time() - t0
        print(f"{out}: {X.shape} in {dt:.0f}s ({len(paths) / dt:.0f} img/s)")


if __name__ == "__main__":
    main()
