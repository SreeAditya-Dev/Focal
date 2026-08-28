"""Package a training-ready copy of the corpus, small enough to upload.

The generated corpus is ~3 GB of 768 px JPEGs, which is awkward to move onto a
hosted GPU runtime. The CNN only ever sees 224x224, and the classical features
were already measured at full resolution and travel as a small table — so a
bundle holding 224 px images plus that table carries everything training needs
at roughly a sixth of the size.

**The trade-off, stated plainly.** Bundled images go through one extra JPEG
round-trip at 224 px that a locally-trained model does not. Quality 98 keeps
that far below the quality-95 encode the corpus already carries, but it is not
nothing, and it lands on the `corruption` class more than the others since that
is the class defined by compression artifacts. The locally-trained model is
therefore the reference; a bundle-trained one should be compared against it
rather than assumed identical.

Use it when moving to a GPU is worth more than that perturbation — which it
usually is, since the alternative is a fine-tuning phase cut short for time.

Usage (from ``ml_pipeline/``)::

    python -m training.export_bundle
    python -m training.export_bundle --quality 100 --zip
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from focal_ml.constants import CNN_INPUT_SIZE


def _resize_one(task: tuple[str, str, str, int]) -> bool:
    import cv2

    cv2.setNumThreads(1)
    relative, source_root, destination_root, quality = task

    source = Path(source_root) / relative
    destination = Path(destination_root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(str(source), dtype=np.uint8)
    if raw.size == 0:
        return False
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return False

    # Matches the training transform exactly: the whole frame squashed to a
    # square, never centre-cropped, so a defect in a corner survives.
    resized = cv2.resize(image, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    buffer.tofile(str(destination))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/bundle"))
    parser.add_argument("--quality", type=int, default=98)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zip", action="store_true", help="also produce bundle.zip")
    args = parser.parse_args()

    import pandas as pd

    manifest_path = args.generated_dir / "manifest.csv"
    features_path = args.generated_dir / "features.parquet"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found")
    if not features_path.exists():
        raise SystemExit(
            f"{features_path} not found — the bundle must carry the features, since they are "
            "measured at 768 px and cannot be recomputed from the 224 px copies."
        )

    manifest = pd.read_csv(manifest_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packing {len(manifest)} images at {CNN_INPUT_SIZE}px (quality {args.quality})...")
    tasks = [
        (path, str(args.generated_dir), str(args.out_dir), args.quality)
        for path in manifest["image_path"]
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.map(_resize_one, tasks, chunksize=32), total=len(tasks)))
        except ImportError:
            results = list(pool.map(_resize_one, tasks, chunksize=32))

    failed = results.count(False)
    if failed:
        raise SystemExit(f"{failed} images could not be packed; the source corpus may be incomplete")

    for name in ("manifest.csv", "features.parquet", "generation_meta.json"):
        source = args.generated_dir / name
        if source.exists():
            shutil.copy(source, args.out_dir / name)

    # Recorded so a bundle-trained checkpoint can never be silently mistaken for
    # one trained on the full-resolution corpus.
    (args.out_dir / "BUNDLE.md").write_text(
        "# Training bundle\n\n"
        f"Images resized to {CNN_INPUT_SIZE}x{CNN_INPUT_SIZE}, JPEG quality {args.quality}.\n\n"
        "Classical features in `features.parquet` were measured on the original "
        "768 px corpus and cannot be recomputed from these copies — several of "
        "them (noise sigma, blockiness, impulse ratio) live in detail that "
        "downscaling destroys, which is the whole reason the model takes them as "
        "a separate input.\n\n"
        "Images here carry one extra JPEG round-trip versus the full corpus. "
        "Compare a model trained from this bundle against the locally-trained "
        "reference rather than assuming they are equivalent.\n",
        encoding="utf-8",
    )

    total_mb = sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"\nWrote {args.out_dir}  ({total_mb:.0f} MB)")

    if args.zip:
        print("Zipping...")
        archive = shutil.make_archive(str(args.out_dir), "zip", str(args.out_dir))
        print(f"Wrote {archive}  ({Path(archive).stat().st_size / 1e6:.0f} MB)")

    print("\nUpload to Drive, then in the Colab notebook set REBUILD=False and point it at the bundle.")


if __name__ == "__main__":
    main()
