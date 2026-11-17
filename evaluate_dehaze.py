"""Run inference and report PSNR/SSIM on the held-out MRSHaze test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from dehazemamba.data import MRSHazeDataset
from dehazemamba.model import DehazeMamba, require_current_checkpoint


def ssim(image: torch.Tensor, target: torch.Tensor) -> float:
    # Gaussian-window SSIM with the standard 11x11, sigma=1.5 settings.
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    kernel_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, 11, 11)
    if min(image.shape[-2:]) < 11:
        raise ValueError("SSIM requires images at least 11x11")
    # Valid windows avoid introducing an artificial black image border.
    mu_x = F.conv2d(image, kernel, groups=3)
    mu_y = F.conv2d(target, kernel, groups=3)
    sigma_x = F.conv2d(image * image, kernel, groups=3) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, kernel, groups=3) - mu_y * mu_y
    sigma_xy = F.conv2d(image * target, kernel, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/johnny/code/dataset/MRSHaze")
    parser.add_argument("--checkpoint", default="outputs/dehazemamba-v2/last.pt")
    parser.add_argument("--output-dir", default="outputs/dehazemamba-v2/evaluation")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS for quick checks")
    parser.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="limit evaluation for a smoke check")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    require_current_checkpoint(checkpoint)
    model_config = checkpoint["model_config"]
    model = DehazeMamba(**model_config).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    perceptual = None
    if not args.no_lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Install lpips to report all paper metrics, or use --no-lpips") from exc
        perceptual = lpips.LPIPS(net=args.lpips_backbone).cuda().eval()

    dataset = MRSHazeDataset(args.data_root, "test")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.no_save_images:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="test", dynamic_ncols=True):
            hazy = batch["hazy"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = model(hazy, sar).float().clamp_(0.0, 1.0)
            for index, name in enumerate(batch["name"]):
                pred = prediction[index : index + 1]
                gt = target[index : index + 1]
                mse = float(torch.mean((pred - gt) ** 2))
                psnr = float("inf") if mse == 0 else -10.0 * float(np.log10(mse))
                image_ssim = ssim(pred, gt)
                row = (name, psnr, image_ssim)
                if perceptual is not None:
                    # LPIPS expects RGB in [-1, 1], and runs in float32.
                    row += (float(perceptual(pred * 2 - 1, gt * 2 - 1).mean()),)
                rows.append(row)
                if not args.no_save_images:
                    array = (
                        pred[0].permute(1, 2, 0).cpu().numpy() * 255.0
                    ).round().clip(0, 255).astype(np.uint8)
                    Image.fromarray(array, mode="RGB").save(output_dir / "predictions" / name)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ("name", "psnr_db", "ssim")
        if perceptual is not None:
            header += (f"lpips_{args.lpips_backbone}",)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow(("mean", *(np.mean([r[i] for r in rows]) for i in range(1, len(header)))))
    lpips_summary = "" if perceptual is None else f" mean_lpips={np.mean([r[3] for r in rows]):.6f}"
    print(
        f"test_count={len(rows)} mean_psnr={np.mean([r[1] for r in rows]):.4f} dB "
        f"mean_ssim={np.mean([r[2] for r in rows]):.6f}{lpips_summary} metrics={csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
