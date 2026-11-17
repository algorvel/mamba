#!/usr/bin/env python3
"""Expose SateHaze1k as MRSHaze-style PNG filename triplets using normalized PNG cache files."""
from __future__ import annotations
import argparse
from PIL import Image, ImageOps
from pathlib import Path

LEVELS = ("Haze1k_thin", "Haze1k_moderate", "Haze1k_thick")
SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def find_image(folder: Path, stem: str) -> Path:
    hits = [p for p in folder.glob(stem + ".*") if p.is_file() and not p.name.startswith(".")]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {stem}.* under {folder}; got {hits}")
    return hits[0]


def normalize_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert(mode)
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR)
        image.save(dst, format="PNG", optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("/home/johnny/code/dataset/SateHaze1k"))
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    base = args.dataset_root / "Haze1k"
    out = args.output_root.resolve()
    counts = {}
    skipped = []
    # This adapter owns its normalized cache; clear old generated images first.
    for split in SPLITS:
        for kind in ("GT", "hazy", "SAR"):
            folder = out / split / kind
            if folder.exists():
                for old in folder.iterdir():
                    if old.is_file() or old.is_symlink():
                        old.unlink()
    for split, split_dir in SPLITS.items():
        paired = 0
        for level in LEVELS:
            root = base / level
            names = ("valid", "val") if split == "valid" else (split_dir,)
            dataset_split = next((root / "dataset" / n for n in names if (root / "dataset" / n).is_dir()), root / "dataset" / split_dir)
            sar_name = "val" if split == "valid" else split
            sar_dir = root / "SAR" / sar_name
            input_dir, target_dir = dataset_split / "input", dataset_split / "target"
            if not input_dir.is_dir() or not target_dir.is_dir() or not sar_dir.is_dir():
                raise FileNotFoundError(f"Missing SateHaze folders for {level}/{split}: {input_dir}, {target_dir}, {sar_dir}")
            inputs = sorted(p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            if not inputs:
                raise FileNotFoundError(f"No input images in {input_dir}")
            sar_files = sorted(p for p in sar_dir.iterdir() if p.is_file() and not p.name.startswith("."))
            if len(sar_files) != len(inputs):
                raise ValueError(f"Unmatched input/SAR counts for {level}/{split}: {len(inputs)} vs {len(sar_files)}")
            for sample_index, hazy in enumerate(inputs):
                stem = hazy.stem
                if stem.endswith("-inputs"):
                    sample_id = stem[:-7]
                else:
                    sample_id = stem
                target_stem = sample_id + "-targets" if stem.endswith("-inputs") else sample_id
                target = find_image(target_dir, target_stem)
                # SAR splits restart numbering at 1 while thin/thick optical
                # validation/test use global IDs; pair the official sorted split order.
                sar = sar_files[sample_index]
                sources = (hazy, target, sar)
                try:
                    for source in sources:
                        with Image.open(source) as image:
                            image.verify()
                except Exception as exc:
                    skipped.append(f"{level}/{split}/{sample_id}: {exc}")
                    continue
                name = f"{level.removeprefix('Haze1k_')}__{sample_id}.png"
                normalize_image(hazy, out / split / "hazy" / name, "RGB")
                normalize_image(target, out / split / "GT" / name, "RGB")
                normalize_image(sar, out / split / "SAR" / name, "L")
                paired += 1
        counts[split] = paired
    print(f"Prepared normalized dataset at {out}: {counts}; skipped_corrupt={len(skipped)}")
    for item in skipped:
        print(f"SKIP {item}")

if __name__ == "__main__":
    main()
