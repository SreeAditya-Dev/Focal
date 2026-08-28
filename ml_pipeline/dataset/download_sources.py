"""Acquire and vet the clean base images the synthetic corpus is built from.

The whole labelling scheme rests on one assumption: the base images are
genuinely clean. A blurry or badly exposed source photo silently poisons the
"no issue" class and every degradation derived from it, so this script does not
just download — it screens every candidate against the same measurements the
model will later be asked to reproduce, and keeps only images that pass.

Usage (from ``ml_pipeline/``)::

    python -m dataset.download_sources --sources coco div2k
    python -m dataset.download_sources --local "C:/my/photos" --target-per-source 500

Output: ``dataset/raw/base_index.json``, consumed by ``generate_synthetic.py``.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from focal_ml.utils import imread_bgr, resize_long_side, to_gray

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    approx_mb: int
    approx_images: int
    note: str


SOURCES: dict[str, Source] = {
    # 100 pristine 2K photographs — the cleanest material available, and small
    # enough to be a reasonable default download.
    "div2k": Source(
        name="div2k",
        url="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        approx_mb=428,
        approx_images=100,
        note="DIV2K validation split, 2K high-resolution",
    ),
    # 800 more of the same, opt-in because of the size.
    "div2k_train": Source(
        name="div2k_train",
        url="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        approx_mb=3300,
        approx_images=800,
        note="DIV2K training split — large download, opt in explicitly",
    ),
    # DIV2K is landscape-heavy; COCO supplies people, interiors, text and
    # close-ups, which is what stops the CNN keying on scene type.
    "coco": Source(
        name="coco",
        url="http://images.cocodataset.org/zips/val2017.zip",
        approx_mb=778,
        approx_images=5000,
        note="COCO val2017 — varied everyday scenes",
    ),
}

DEFAULT_SOURCES = ("div2k", "coco")

# --------------------------------------------------------------------------
# Screening thresholds
#
# All measured on a 512px-long-side copy so the numbers mean the same thing
# regardless of the source's native resolution.
# --------------------------------------------------------------------------

SCREEN_LONG_SIDE = 512
MIN_WIDTH, MIN_HEIGHT = 640, 480
MIN_SHARPNESS = 150.0        # variance of Laplacian
BRIGHTNESS_RANGE = (70.0, 185.0)   # mean luma — excludes already-mis-exposed shots
MIN_CONTRAST = 25.0          # RMS contrast — excludes flat/hazy frames
MAX_NOISE_SIGMA = 4.0        # estimated sensor noise, 0-255 scale

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial download if present."""
    import requests  # imported lazily so the module loads without the [data] extra

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    partial = dest.with_suffix(dest.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        if have and response.status_code == 200:
            have = 0  # server ignored the range request; start over
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) + have

        try:
            from tqdm import tqdm
            bar = tqdm(total=total or None, initial=have, unit="B", unit_scale=True, desc=f"  {dest.name}")
        except ImportError:
            bar = None

        mode = "ab" if have else "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=chunk):
                handle.write(block)
                if bar is not None:
                    bar.update(len(block))
        if bar is not None:
            bar.close()

    partial.rename(dest)
    return dest


def _extract(archive: Path, dest: Path) -> Path:
    """Extract a zip once; a marker file makes reruns cheap."""
    marker = dest / ".extracted"
    if marker.exists():
        print(f"  already extracted: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    marker.touch()
    return dest


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------


@dataclass
class BaseImage:
    path: str
    source: str
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    noise_sigma: float


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate: convolve with a kernel that annihilates
    smooth content and locally-linear gradients, leaving mostly noise."""
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * (width - 2) * (height - 2)))


def screen_image(path: Path, source: str) -> BaseImage | None:
    """Measure one candidate; return it only if it qualifies as clean."""
    image = imread_bgr(path)
    if image is None:
        return None
    height, width = image.shape[:2]
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return None

    small = resize_long_side(image, SCREEN_LONG_SIDE)
    gray = to_gray(small)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    noise_sigma = _estimate_noise_sigma(gray)

    if sharpness < MIN_SHARPNESS:
        return None
    if not (BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]):
        return None
    if contrast < MIN_CONTRAST:
        return None
    if noise_sigma > MAX_NOISE_SIGMA:
        return None

    return BaseImage(
        path=str(path.resolve()),
        source=source,
        width=width,
        height=height,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        noise_sigma=round(noise_sigma, 4),
    )


def _screen_one(args: tuple[str, str]) -> dict | None:
    path, source = args
    result = screen_image(Path(path), source)
    return asdict(result) if result is not None else None


def screen_directory(
    root: Path, source: str, target: int, workers: int, rng: np.random.Generator
) -> list[dict]:
    """Screen every image under ``root``, keeping up to ``target`` that pass."""
    candidates = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not candidates:
        print(f"  no images found under {root}")
        return []

    # Shuffle before screening so a cap of N does not just take the alphabetically
    # first N, which in COCO correlates with capture batch.
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    print(f"  screening {len(candidates)} candidates from {source} (keeping up to {target})")
    kept: list[dict] = []
    payload = [(str(p), source) for p in candidates]

    if workers <= 1:
        for item in payload:
            result = _screen_one(item)
            if result is not None:
                kept.append(result)
            if len(kept) >= target:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_screen_one, item): item for item in payload}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        kept.append(result)
                    if len(kept) >= target:
                        break
            finally:
                for future in futures:
                    future.cancel()

    rejected = len(candidates) - len(kept)
    print(f"  kept {len(kept)}, rejected or unused {rejected}")
    return kept[:target]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sources", nargs="*", default=list(DEFAULT_SOURCES), choices=sorted(SOURCES),
        help="public datasets to download and screen",
    )
    parser.add_argument(
        "--local", type=Path, default=None,
        help="also screen your own clean images from this directory (no download)",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--target-per-source", type=int, default=1500,
        help="cap on images kept per source after screening",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--keep-archives", action="store_true",
        help="keep the downloaded .zip files (default: delete after extraction)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    planned_mb = sum(SOURCES[s].approx_mb for s in args.sources)
    if planned_mb:
        print(f"Planned download: ~{planned_mb} MB across {len(args.sources)} source(s)\n")

    records: list[dict] = []

    for name in args.sources:
        source = SOURCES[name]
        print(f"[{name}] {source.note}")
        archive = _download(source.url, raw_dir / f"{name}.zip")
        extracted = _extract(archive, raw_dir / name)
        records.extend(screen_directory(extracted, name, args.target_per_source, args.workers, rng))
        if not args.keep_archives and archive.exists():
            archive.unlink()
            print(f"  removed archive {archive.name}")
        print()

    if args.local is not None:
        print(f"[local] {args.local}")
        records.extend(screen_directory(args.local, "local", args.target_per_source, args.workers, rng))
        print()

    if not records:
        raise SystemExit(
            "No images passed screening. Loosen --target-per-source or the thresholds "
            "at the top of this file, or check that the source directories contain images."
        )

    index_path = raw_dir / "base_index.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "screening": {
                    "screen_long_side": SCREEN_LONG_SIDE,
                    "min_resolution": [MIN_WIDTH, MIN_HEIGHT],
                    "min_sharpness_laplacian_var": MIN_SHARPNESS,
                    "brightness_mean_range": list(BRIGHTNESS_RANGE),
                    "min_rms_contrast": MIN_CONTRAST,
                    "max_noise_sigma": MAX_NOISE_SIGMA,
                },
                "counts": {
                    source: sum(1 for r in records if r["source"] == source)
                    for source in sorted({r["source"] for r in records})
                },
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} screened base images to {index_path}")
    for source in sorted({r["source"] for r in records}):
        count = sum(1 for r in records if r["source"] == source)
        print(f"  {source:<12} {count}")
    print("\nNext: python -m dataset.generate_synthetic")


if __name__ == "__main__":
    main()
