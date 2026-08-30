"""Generate synthetic defective images via ControlNet (Canny) + Stable
Diffusion inpainting, from a category's train/good images.

MVTec-AD's train/ split only contains "good" images, so the baseline
classifier and defect-type classifier both have to be trained on splits
carved out of the labeled test/ set instead (see README's "misleadingly
perfect classifier" finding). This script generates genuinely novel,
train-only defective examples instead: it Canny-edges a pristine image to
preserve its geometry, masks a small randomly-placed region, and inpaints
a text-prompted defect only inside that mask -- producing a synthetic
defective image with an exact ground-truth mask, never derived from
test/.

Requires the optional heavy dependencies (not part of the main pipeline):
    pip install -r requirement-synthetic.txt

Usage:
    python -m src.data.generate_synthetic_defects --category screw \
        --defect-type synthetic_rust --prompt "deep rust, corrosion damage" \
        --num-images 10
"""

import argparse
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "synthetic_defects"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
MANIFEST_COLUMNS = ["image_path", "mask_path", "split", "label", "defect_type"]

NEGATIVE_PROMPT = "cartoon, illustration, changing shape, out of frame, blurry, low quality"


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_random_mask(size: int, mask_mode: str, rng: random.Random) -> Image.Image:
    """A single-channel mask marking the region to inpaint. `mask_mode`
    controls the defect's rough shape -- 'scratch' (a thick diagonal
    line), 'blob' (an ellipse), or 'patch' (a rectangle) -- all randomly
    placed and sized so repeated calls produce varied defect geometry and
    location. Pure function (no model dependency), so it's independently
    testable without the diffusers/transformers/accelerate stack."""
    if mask_mode not in ("scratch", "blob", "patch"):
        raise ValueError(f"Unsupported mask_mode '{mask_mode}'")

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx = rng.randint(size // 4, 3 * size // 4)
    cy = rng.randint(size // 4, 3 * size // 4)

    if mask_mode == "scratch":
        length = rng.randint(size // 4, size // 2)
        angle = rng.uniform(0, 2 * np.pi)
        dx, dy = int(length * np.cos(angle) / 2), int(length * np.sin(angle) / 2)
        width = rng.randint(max(1, size // 40), max(2, size // 15))
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=255, width=width)
    elif mask_mode == "blob":
        rx, ry = rng.randint(size // 20, size // 8), rng.randint(size // 20, size // 8)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    else:  # patch
        half = rng.randint(size // 20, size // 8)
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=255)

    return mask


def make_canny_control_image(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Edge map used to condition ControlNet, preserving the source
    image's physical geometry so the inpainted defect doesn't distort the
    object's actual shape."""
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def _manifest_path(path: Path) -> str:
    """Path stored in the manifest CSV: relative to PROJECT_ROOT when
    possible (matching create_manifest.py's convention), falling back to
    an absolute path if `path` sits outside the project (e.g. a custom
    --output-root, or a test's tmp_path)."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_pipeline(device: str):
    """Loads the ControlNet + Stable-Diffusion-inpainting pipeline. Kept
    as a separate, lazily-imported function so the rest of this module
    (mask generation, manifest writing) stays importable/testable without
    the heavy diffusers/transformers/accelerate dependencies installed."""
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        UniPCMultistepScheduler,
    )

    # fp16 halves memory/time on CUDA; MPS/CPU don't reliably support fp16
    # for every op these pipelines use, so stick to fp32 there.
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe


def generate_synthetic_defects(
    category: str,
    defect_type: str,
    prompt: str,
    num_images: int,
    mask_mode: str,
    image_size: int,
    controlnet_conditioning_scale: float,
    num_inference_steps: int,
    seed: int,
    data_root: Path,
    output_root: Path,
    pipeline_factory=build_pipeline,
) -> List[dict]:
    """Generates `num_images` synthetic defective images for `category`
    and returns manifest-schema rows for them. `pipeline_factory` is
    injectable so tests can substitute a stub pipeline without loading
    the real Stable Diffusion weights."""
    device = resolve_device()
    print(f"Using device: {device}")

    train_good_dir = data_root / category / "train" / "good"
    source_paths = sorted(train_good_dir.glob("*.png"))[:num_images]
    if not source_paths:
        raise FileNotFoundError(f"No train/good images found under {train_good_dir}")

    image_out_dir = output_root / category / defect_type
    mask_out_dir = output_root / category / "ground_truth" / defect_type
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_factory(device)
    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)
    rng = random.Random(seed)

    rows = []
    for i, source_path in enumerate(source_paths):
        print(f"[{i + 1}/{len(source_paths)}] Generating synthetic defect from {source_path.name}...")
        init_image = Image.open(source_path).convert("RGB").resize((image_size, image_size))
        control_image = make_canny_control_image(init_image)
        mask_image = make_random_mask(image_size, mask_mode, rng)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        synthetic_image = result.images[0]

        out_stem = f"synthetic_{i:03d}_{source_path.stem}"
        image_out_path = image_out_dir / f"{out_stem}.png"
        mask_out_path = mask_out_dir / f"{out_stem}_mask.png"
        synthetic_image.save(image_out_path)
        mask_image.save(mask_out_path)

        rows.append(
            {
                "image_path": _manifest_path(image_out_path),
                "mask_path": _manifest_path(mask_out_path),
                "split": "train",
                "label": 1,
                "defect_type": defect_type,
            }
        )

    return rows


def write_synthetic_manifest(rows: List[dict], category: str, manifest_dir: Path) -> Path:
    """Writes/appends to data/manifests/<category>_synthetic.csv -- a
    manifest-schema-compatible CSV kept separate from the real
    data/manifests/<category>.csv so synthetic images never get mistaken
    for real MVTec-AD data or silently mixed into the canonical manifest.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"{category}_synthetic.csv"
    new_rows = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--defect-type",
        required=True,
        help="Label for this synthetic defect batch, e.g. 'synthetic_rust'.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt describing the defect to inpaint.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="How many train/good source images to generate defects from.",
    )
    parser.add_argument("--mask-mode", choices=("scratch", "blob", "patch"), default="scratch")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.8)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_synthetic_defects(
        category=args.category,
        defect_type=args.defect_type,
        prompt=args.prompt,
        num_images=args.num_images,
        mask_mode=args.mask_mode,
        image_size=args.image_size,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        data_root=args.data_root,
        output_root=args.output_root,
    )
    manifest_path = write_synthetic_manifest(rows, args.category, args.manifest_dir)
    print(f"Generated {len(rows)} synthetic defective images under {args.output_root / args.category}")
    print(f"Wrote/updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
