"""Precompute frozen-backbone embeddings for the head-training phase.

While the backbone is frozen its 576-d output is a fixed function of the image,
so every epoch recomputes the same convolutions — and on CPU that convolution is
essentially the whole cost of a training step. Computing them once turns Phase A
from roughly 40 minutes per epoch into under one, and the cache is reused by
every ablation run.

Both the unflipped and horizontally flipped embedding are stored, so the head
still sees the flip augmentation it would have had. Flipping is the only
augmentation in this pipeline, so caching both covers it exactly rather than
approximating it.

The cache is only valid while the backbone is frozen. Phase B reopens the last
blocks, the embeddings change every step, and training falls back to running the
network properly.

Usage (from ``ml_pipeline/``)::

    python -m training.cache_embeddings --split train
    python -m training.cache_embeddings --split val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from focal_ml.model.architecture import FocalNet, ModelConfig
from training.dataset import load_split


def compute_embeddings(
    model: FocalNet, loader: DataLoader, device: torch.device, flip: bool
) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if flip:
                # dims=[3] is the width axis of an (N, C, H, W) batch.
                images = torch.flip(images, dims=[3])
            chunks.append(model.embed_image(images).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/generated/embeddings"))
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pretrained weights are what gets cached: this runs before training, and
    # Phase A never modifies the backbone, so the ImageNet initialisation is
    # exactly the frozen trunk the heads will be trained against.
    model = FocalNet(ModelConfig(use_features=False, pretrained=True)).to(device)
    model.freeze_backbone()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   batch: {args.batch_size}")

    for split in args.splits:
        dataset = load_split(args.generated_dir, split, with_features=False, load_image=True)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )

        for flip in (False, True):
            name = f"{split}{'_flip' if flip else ''}.npy"
            destination = args.out_dir / name
            if destination.exists():
                print(f"  {name}: already cached")
                continue

            started = time.perf_counter()
            embeddings = compute_embeddings(model, loader, device, flip)
            np.save(destination, embeddings)
            elapsed = time.perf_counter() - started
            print(
                f"  {name}: {embeddings.shape} in {elapsed / 60:.1f} min "
                f"({elapsed / max(len(dataset), 1) * 1000:.0f} ms/image)"
            )

        # The row order is the dataset's path order; anything reading the cache
        # must line up against the same list or every image trains on another
        # image's embedding.
        (args.out_dir / f"{split}_paths.txt").write_text(
            "\n".join(dataset.paths), encoding="utf-8"
        )

    print(f"\nWrote {args.out_dir}")
    print("Train with: python -m training.train --embeddings dataset/generated/embeddings")


if __name__ == "__main__":
    main()
