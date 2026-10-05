"""Controlled image degradations applied to patches before embedding.

A condition is the string "clean" or "<kind>:<level>", e.g. "blur:2", "jpeg:30", "stain:0.2".
"""
import io

import numpy as np
from PIL import Image, ImageFilter
from skimage.color import hed2rgb, rgb2hed

# Levels run from mild to severe for every kind.
LEVELS = {
    "blur": [0.5, 1, 2, 3, 3.5, 4, 5, 6],  # Gaussian sigma in pixels (focus blur proxy)
    "jpeg": [70, 50, 30, 25, 20, 15, 10, 5],  # JPEG quality factor
    "stain": [0.1, 0.2, 0.3, 0.4, 0.5], # relative H/E imbalance
}


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def jpeg(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def stain(img: Image.Image, alpha: float) -> Image.Image:
    """Deterministic stain shift: strengthen haematoxylin, weaken eosin by `alpha`."""
    hed = rgb2hed(np.asarray(img))
    hed[..., 0] *= 1 + alpha
    hed[..., 1] *= 1 - alpha
    rgb = np.clip(hed2rgb(hed), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


_FUNCS = {"blur": blur, "jpeg": jpeg, "stain": stain}


def apply(img: Image.Image, condition: str) -> Image.Image:
    if condition == "clean":
        return img
    kind, level = condition.split(":")
    return _FUNCS[kind](img, float(level))


def all_conditions() -> list[str]:
    return ["clean"] + [f"{k}:{v}" for k, levels in LEVELS.items() for v in levels]


def tag(condition: str) -> str:
    """Filesystem-safe name for a condition."""
    return condition.replace(":", "_")
