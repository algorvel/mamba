"""Train DehazeMamba DM-T on MRSHaze."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dehazemamba.data import MRSHazeDataset
from dehazemamba.losses import dehaze_loss, InnovationLoss
from dehazemamba.model import ARCHITECTURE_VERSION, MODEL_DEPTHS, DehazeMamba, require_current_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/johnny/code/dataset/MRSHaze")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-size", choices=MODEL_DEPTHS, default="T")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment", choices=("baseline", "innovate-1"), default="baseline")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--wavelet-weight", type=float, default=0.1)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", default=None)
    start.add_argument("--init-checkpoint", default=None, help="initialize from a v2 baseline; fresh optimizer")
    parser.add_argument("--max-steps", type=int, default=None, help="debug smoke run only")
    return parser.parse_args()


def save_checkpoint(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested reproduction run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_set = MRSHazeDataset(args.data_root, "train")
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    innovation = args.experiment == "innovate-1"
    model_config = dict(base_channels=args.base_channels, state_dim=args.state_dim,
                        depths=list(MODEL_DEPTHS[args.model_size]),
                        architecture_version=ARCHITECTURE_VERSION, wavelet_refinement=innovation)
    loss_config = ({"name": "innovate-1", "mse_weight": args.mse_weight,
                    "wavelet_weight": args.wavelet_weight, "levels": 2, "r": 0.4}
                   if innovation else {"name": "baseline", "frequency_weight": 0.1})
    criterion = InnovationLoss(args.mse_weight, args.wavelet_weight).cuda() if innovation else None
    args.output_dir = args.output_dir or ("outputs/innovate-1" if innovation else "outputs/dehazemamba-v2")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if (output_dir / "last.pt").exists() and not (
        args.resume and Path(args.resume).expanduser().resolve().parent == output_dir
    ):
        raise FileExistsError(f"Use a new output directory; checkpoint already exists in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = DehazeMamba(**model_config).cuda()
    if args.init_checkpoint:
        if not innovation:
            raise ValueError("--init-checkpoint is for baseline-to-innovate-1 initialization")
        initial = torch.load(args.init_checkpoint, map_location="cuda", weights_only=False)
        require_current_checkpoint(initial)
        expected = {**model_config, "wavelet_refinement": False}
        if {"wavelet_refinement": False, **initial["model_config"]} != expected:
            raise ValueError("Initialization checkpoint must be a matching v2 baseline")
        # Merge only the new zero-initialized refinement parameters; load all keys strictly.
        state = {k: v for k, v in model.state_dict().items() if k.startswith("refinement.")}
        state.update(initial["model"])
        model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        require_current_checkpoint(checkpoint)
        if {"wavelet_refinement": False, **checkpoint["model_config"]} != model_config:
            raise ValueError("Resume model configuration differs; use matching architecture flags")
        saved_loss = checkpoint.get("loss_config", {"name": "baseline", "frequency_weight": 0.1})
        if saved_loss != loss_config:
            raise ValueError("Resume loss configuration differs; use matching experiment and weights")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        print(f"Resumed from epoch {start_epoch}, step {global_step}", flush=True)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if args.resume and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"AMP={amp_dtype} train={len(train_set)} batch={args.batch_size} "
        f"parameters={n_params:,} epochs={args.epochs}",
        flush=True,
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        component_sums = {}
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            hazy = batch["hazy"].cuda(non_blocking=True)
            sar = batch["sar"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = model(hazy, sar)
            if innovation:
                loss, components = criterion(prediction, target)
            else:
                loss, spatial, frequency = dehaze_loss(prediction.float(), target)
                components = {"l1": spatial, "fft": frequency}
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value)
            global_step += 1
            progress.set_postfix(
                loss=f"{float(loss.detach()):.5f}",
                **{name: f"{float(value):.5f}" for name, value in components.items()},
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if args.max_steps and global_step >= args.max_steps:
                break

        scheduler.step()
        mean_loss = running_loss / max(1, min(len(loader), batch_index + 1))
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "mean_loss": mean_loss,
            "model_config": model_config,
            "loss_config": loss_config,
            "loss_components": {k: v / (batch_index + 1) for k, v in component_sums.items()},
            "train_config": vars(args),
        }
        save_checkpoint(output_dir / "last.pt", payload)
        print(
            f"epoch={epoch + 1} mean_loss={mean_loss:.6f} "
            f"components={payload['loss_components']} "
            f"elapsed_hours={(time.time() - start_time) / 3600:.2f} "
            f"checkpoint={output_dir / 'last.pt'}",
            flush=True,
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Training stopped after {global_step} optimizer steps", flush=True)


if __name__ == "__main__":
    main()
