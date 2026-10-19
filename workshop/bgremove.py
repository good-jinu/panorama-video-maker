"""Background removal backends (ONNX Runtime).

Models (pick with `--model` or $WORKSHOP_BG_MODEL):
  withoutbg          withoutBG open weights: router -> coarse/BiRefNet pipeline (default)
  birefnet           BiRefNet general, 1024px, MIT. Best edges on fur/hair; ~1 GB download
  birefnet-lite      BiRefNet lite (Swin-T), 1024px, MIT. ~4x smaller/faster
  birefnet-portrait  BiRefNet trained for people/portraits, MIT
  rmbg2              BRIA RMBG-2.0 (BiRefNet arch). Gated + CC BY-NC 4.0: accept the license on
                     huggingface.co/briaai/RMBG-2.0 and set HF_TOKEN. Non-commercial use only.

Execution providers (pick with $WORKSHOP_ORT_PROVIDERS, comma-separated):
  auto (default) -> CoreML (Apple Neural Engine/GPU) with CPU fallback when available, else CPU
  cpu, coreml, or explicit ORT names like "CoreMLExecutionProvider,CPUExecutionProvider"
"""
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

WITHOUTBG_REPO = "withoutbg/withoutbg-openweights-onnx"
REPO_ID = WITHOUTBG_REPO  # backwards-compatible alias

# name -> (repo_id, onnx file, input size)
BIREFNET_MODELS: dict[str, tuple[str, str, int]] = {
    "birefnet": ("onnx-community/BiRefNet-ONNX", "onnx/model.onnx", 1024),
    "birefnet-lite": ("onnx-community/BiRefNet_lite-ONNX", "onnx/model.onnx", 1024),
    "birefnet-portrait": ("onnx-community/BiRefNet-portrait-ONNX", "onnx/model.onnx", 1024),
    "rmbg2": ("yamura4/RMBG-2.0-ONNX", "onnx/model.onnx", 1024),
}
MODEL_NAMES = ("withoutbg", *BIREFNET_MODELS)
DEFAULT_MODEL = "withoutbg"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CACHE_DIR = Path.home() / ".cache" / "image-generating" / "coreml"


def resolve_providers(spec: str | Sequence[str] | None = None) -> list:
    """Turn 'auto' / 'cpu' / 'coreml' / explicit ORT names into an ORT providers list."""
    if spec is None:
        spec = os.environ.get("WORKSHOP_ORT_PROVIDERS", "cpu")
    names = [s.strip() for s in spec.split(",")] if isinstance(spec, str) else list(spec)
    available = ort.get_available_providers()
    aliases = {"cpu": "CPUExecutionProvider", "coreml": "CoreMLExecutionProvider", "cuda": "CUDAExecutionProvider"}

    if names == ["auto"]:
        names = ["CPUExecutionProvider"]
    names = [aliases.get(n.lower(), n) for n in names]
    if "CPUExecutionProvider" not in names:
        names.append("CPUExecutionProvider")

    providers = []
    for name in names:
        if name not in available:
            raise RuntimeError(f"ONNX Runtime provider '{name}' unavailable (have: {available})")
        if name == "CoreMLExecutionProvider":
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            providers.append((name, {
                "ModelFormat": "MLProgram",  # supports far more ops than the legacy NeuralNetwork format
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
                "ModelCacheDirectory": str(_CACHE_DIR),  # avoid recompiling the model every run
            }))
        else:
            providers.append(name)
    return providers


def _session(path: Path | str, providers: list) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # hide CoreML partitioning warnings
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _bilinear(x, height, width):
    """(C, H, W) -> (C, height, width); torch bilinear, align_corners=False, no antialias."""
    def axis(src, dst):
        pos = np.maximum((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0.0)
        i0 = pos.astype(np.int64)
        return i0, np.minimum(i0 + 1, src - 1), (pos - i0).astype(np.float32)

    _, h, w = x.shape
    r0, r1, rl = axis(h, height)
    c0, c1, cl = axis(w, width)
    rows = x[:, r0] * (1 - rl)[:, None] + x[:, r1] * rl[:, None]
    return rows[:, :, c0] * (1 - cl) + rows[:, :, c1] * cl


def _feed(rgb, spec):
    feeds = {}
    for name, s in spec["inputs"].items():
        if s["interpolation"] == "bicubic":  # antialiased; PIL "F" mode matches torch
            x = np.stack([
                np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]))
                           .resize((s["size"],) * 2, Image.Resampling.BICUBIC))
                for c in range(3)
            ])
        else:
            x = _bilinear(np.ascontiguousarray(rgb.transpose(2, 0, 1)), s["size"], s["size"])
        feeds[name] = np.ascontiguousarray(x[None], dtype=np.float32)
    return feeds


class _Remover:
    name: str

    def matte(self, image: Image.Image) -> tuple[Image.Image, str]:
        raise NotImplementedError

    def remove(self, image: Image.Image) -> tuple[Image.Image, str]:
        """Return (RGBA cutout, route/model label)."""
        matte, label = self.matte(image)
        out = image.convert("RGB")
        out.putalpha(matte)
        return out, label


class WithoutBGRemover(_Remover):
    """withoutBG open weights: a router picks the coarse or BiRefNet branch per image."""

    name = "withoutbg"

    def __init__(self, providers: list | None = None):
        providers = providers or resolve_providers()
        bundle = Path(snapshot_download(WITHOUTBG_REPO, allow_patterns=["*.onnx", "*.onnx.json"]))
        self.sidecar = json.loads((bundle / "withoutbg-open-weights.onnx.json").read_text())
        self.sessions = {
            name: _session(bundle / self.sidecar[name]["file"], providers)
            for name in ("router", "coarse", "birefnet")
        }

    def matte(self, image: Image.Image) -> tuple[Image.Image, str]:
        image = image.convert("RGB")
        sc = self.sidecar
        w, h = image.size
        max_w, max_h = sc["max_inference_size"]
        if w > max_w:
            w, h = max_w, int(max_w / (image.width / image.height))
        if h > max_h:
            w, h = int(max_h * (image.width / image.height)), max_h
        work = image.resize((w, h)) if (w, h) != image.size else image
        rgb = np.asarray(work, dtype=np.float32) / 255.0

        router = sc["router"]
        router_feeds = _feed(rgb, router)
        logits, *feats = self.sessions["router"].run(
            [router["logits_name"], *router["feature_names"]], router_feeds)
        category = router["categories"][int(np.argmax(logits[0]))]

        if category in router["birefnet_categories"]:
            spec = sc["birefnet"]
            alpha = self.sessions["birefnet"].run([spec["output_name"]], _feed(rgb, spec))[0][0]
        else:
            spec = sc["coarse"]
            feeds = {**router_feeds, **_feed(rgb, {"inputs": {"rgb_depth": spec["inputs"]["rgb_depth"]}})}
            feeds.update(zip(spec["feature_names"], feats))
            alpha = self.sessions["coarse"].run([spec["output_name"]], feeds)[0][0]

        alpha = np.clip(_bilinear(alpha, h, w), 0.0, 1.0)[0]
        matte = Image.fromarray(np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
        return matte.resize(image.size, Image.Resampling.BILINEAR), category


class BiRefNetRemover(_Remover):
    """Single-pass BiRefNet-family model (BiRefNet, BiRefNet-lite/portrait, RMBG-2.0)."""

    def __init__(self, model: str = "birefnet", providers: list | None = None):
        if model not in BIREFNET_MODELS:
            raise ValueError(f"Unknown BiRefNet model '{model}' (expected one of {list(BIREFNET_MODELS)})")
        self.name = model
        repo_id, filename, self.size = BIREFNET_MODELS[model]
        path = hf_hub_download(repo_id, filename)
        self.session = _session(path, providers or resolve_providers())
        self.input_name = self.session.get_inputs()[0].name

    def matte(self, image: Image.Image) -> tuple[Image.Image, str]:
        image = image.convert("RGB")
        work = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = (np.asarray(work, dtype=np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        raw = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        # Some ONNX exports (like BiRefNet) output logits, while others (like RMBG-2.0) already output sigmoid probabilities [0, 1]
        if raw.min() < 0.0 or raw.max() > 1.0:
            alpha = 1.0 / (1.0 + np.exp(-raw[0, 0]))
        else:
            alpha = raw[0, 0]
        matte = Image.fromarray(np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8))
        return matte.resize(image.size, Image.Resampling.BILINEAR), self.name


def BackgroundRemover(model: str | None = None, providers: str | Sequence[str] | None = None):
    """Factory: build the remover for `model` (default: $WORKSHOP_BG_MODEL or 'withoutbg')."""
    model = (model or os.environ.get("WORKSHOP_BG_MODEL") or DEFAULT_MODEL).lower()
    resolved = resolve_providers(providers)
    if model == "withoutbg":
        return WithoutBGRemover(resolved)
    return BiRefNetRemover(model, resolved)
