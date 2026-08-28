"""Build the labelled synthetic corpus from screened clean base images.

For every base image this writes a family of variants — clean, one per issue
type, and a few realistic multi-issue combinations — and records exact
ground-truth labels for each.

Two decisions matter for the validity of the experiment:

  * **Splits are assigned per base image, before degradation.** Every variant of
    a scene lands in the same split. Splitting afterwards would put a blurred
    and a noisy copy of the same photograph on both sides of the train/test
    boundary, and the resulting scores would measure memorisation.
  * **Images are resized to the canonical long side before degradation**, not
    after. Downscaling a noisy image averages the noise away; the degradation
    must be applied at the resolution the model will actually observe.

Usage (from ``ml_pipeline/``)::

    python -m dataset.generate_synthetic                 # full corpus
    python -m dataset.generate_synthetic --limit 20      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from focal_ml.constants import (
    CANONICAL_LONG_SIDE,
    GENERATED_JPEG_QUALITY,
    ISSUE_TYPES,
    compute_quality_score,
    quality_label,
    severity_bucket_from_score,
)
from focal_ml.utils import imread_bgr, imwrite_jpeg, jpeg_roundtrip, resize_long_side

from dataset.degradations import APPLICATION_ORDER, DegradationResult, apply_degradation

SPLIT_RATIOS = (0.70, 0.15, 0.15)

#: Degradation pairs that co-occur in real capture. Sampling these more often
#: than uniform random pairs keeps the multi-issue examples physically sensible:
#: a dim scene forces high ISO (noise) and a slow shutter (blur), and heavily
#: reshared images are both noisy and block-corrupted.
PLAUSIBLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("underexposure", "noise"),
    ("underexposure", "blur"),
    ("blur", "noise"),
    ("noise", "corruption"),
    ("blur", "corruption"),
    ("overexposure", "blur"),
    ("defect", "noise"),
    ("defect", "blur"),
    ("overexposure", "corruption"),
)

#: Physically contradictory — an image cannot be both too dark and too bright.
FORBIDDEN_PAIRS: frozenset[frozenset[str]] = frozenset(
    {frozenset({"underexposure", "overexposure"})}
)

#: Prior over severity buckets (low, medium, high). Deliberately not uniform:
#: among real photographs that have a problem at all, mild problems vastly
#: outnumber catastrophic ones. Sampling uniformly pushes roughly half the
#: corpus into the POOR band and leaves the model over-exposed to extremes it
#: will rarely be asked about.
SEVERITY_PRIOR: tuple[float, float, float] = (0.45, 0.35, 0.20)


@dataclass(frozen=True)
class GenConfig:
    """Picklable settings passed to each worker process."""

    out_dir: str
    long_side: int = CANONICAL_LONG_SIDE
    jpeg_quality: int = GENERATED_JPEG_QUALITY
    clean_variants: int = 2
    combo_variants: int = 2
    combo_max_bucket: int = 2
    seed: int = 20240828


# --------------------------------------------------------------------------
# Manifest rows
# --------------------------------------------------------------------------


def _blank_labels() -> dict:
    labels: dict = {}
    for issue in ISSUE_TYPES:
        labels[f"{issue}_present"] = 0
        labels[f"{issue}_severity_bucket"] = 0
        labels[f"{issue}_severity_score"] = 0.0
    return labels


def _row(
    *,
    image_path: str,
    base_image_id: str,
    source: str,
    split: str,
    variant: str,
    results: list[DegradationResult],
    width: int,
    height: int,
) -> dict:
    """Assemble one manifest row from the degradations actually applied."""
    labels = _blank_labels()
    severities: dict[str, float] = {}
    provenance = []

    for result in results:
        issue = result.issue_type
        # If the same issue were applied twice, the visible severity is the
        # stronger of the two rather than their sum.
        score = max(severities.get(issue, 0.0), result.severity_score)
        severities[issue] = score
        labels[f"{issue}_present"] = 1
        labels[f"{issue}_severity_score"] = round(score, 4)
        labels[f"{issue}_severity_bucket"] = severity_bucket_from_score(score)
        provenance.append(
            {
                "issue": issue,
                "method": result.method,
                "severity_score": round(result.severity_score, 4),
                "params": result.params,
            }
        )

    score = compute_quality_score(severities)
    return {
        "image_path": image_path,
        "base_image_id": base_image_id,
        "source": source,
        "split": split,
        "variant": variant,
        "is_clean": int(not results),
        "n_issues": len(severities),
        **labels,
        "quality_score": score,
        "quality_label": quality_label(score),
        "width": width,
        "height": height,
        "degradation_params": json.dumps(provenance, separators=(",", ":")),
    }


# --------------------------------------------------------------------------
# Per-image generation
# --------------------------------------------------------------------------


def _sample_bucket(rng: np.random.Generator, max_bucket: int = 3) -> int:
    """Draw a severity bucket from ``SEVERITY_PRIOR``, capped at ``max_bucket``."""
    weights = np.array(SEVERITY_PRIOR[:max_bucket], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, max_bucket + 1), p=weights))


def _choose_pair(rng: np.random.Generator) -> tuple[str, str]:
    """Pick two co-occurring issue types, favouring plausible combinations."""
    if rng.random() < 0.7:
        index = int(rng.integers(0, len(PLAUSIBLE_PAIRS)))
        return PLAUSIBLE_PAIRS[index]
    while True:
        pair = tuple(rng.choice(ISSUE_TYPES, size=2, replace=False))
        if frozenset(pair) not in FORBIDDEN_PAIRS:
            return pair  # type: ignore[return-value]


def _sanitise(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40]


def generate_for_base(task: tuple[int, dict, str, GenConfig]) -> list[dict]:
    """Produce every variant of one base image. Runs inside a worker process."""
    index, record, split, config = task

    image = imread_bgr(record["path"])
    if image is None:
        return []
    image = resize_long_side(image, config.long_side)
    height, width = image.shape[:2]

    source = record["source"]
    base_id = f"{source}_{index:05d}_{_sanitise(Path(record['path']).stem)}"
    out_root = Path(config.out_dir)
    rows: list[dict] = []

    # Seeding from (global seed, image index) makes generation reproducible and
    # independent of how work is distributed across worker processes.
    rng = np.random.default_rng([config.seed, index])

    def emit(variant: str, img: np.ndarray, results: list[DegradationResult]) -> None:
        relative = f"{split}/{base_id}__{variant}.jpg"
        if not imwrite_jpeg(out_root / relative, img, config.jpeg_quality):
            return
        rows.append(
            _row(
                image_path=relative,
                base_image_id=base_id,
                source=source,
                split=split,
                variant=variant,
                results=results,
                width=img.shape[1],
                height=img.shape[0],
            )
        )

    # --- clean ---------------------------------------------------------
    emit("clean0", image, [])
    for i in range(1, config.clean_variants):
        # Benign transforms only. These stay labelled clean on purpose: the model
        # must learn that a horizontal flip and ordinary high-quality compression
        # are not defects, or it will flag every re-saved photograph.
        variant_img = np.ascontiguousarray(image[:, ::-1]) if rng.random() < 0.5 else image
        variant_img = jpeg_roundtrip(variant_img, int(rng.integers(88, 97)))
        emit(f"clean{i}", variant_img, [])

    # --- one variant per issue type ------------------------------------
    for issue in ISSUE_TYPES:
        result = apply_degradation(image, issue, _sample_bucket(rng), rng)
        emit(issue, result.image, [result])

    # --- multi-issue combinations --------------------------------------
    for i in range(config.combo_variants):
        pair = _choose_pair(rng)
        ordered = sorted(pair, key=APPLICATION_ORDER.index)
        current = image
        results: list[DegradationResult] = []
        for issue in ordered:
            # Capped severity: two independent severe defects on one frame is
            # possible but rare, and would dominate the loss if generated freely.
            bucket = _sample_bucket(rng, config.combo_max_bucket)
            result = apply_degradation(current, issue, bucket, rng)
            current = result.image
            results.append(result)
        emit(f"combo{i}_{'_'.join(ordered)}", current, results)

    return rows


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def assign_splits(records: list[dict], seed: int) -> list[str]:
    """Assign each base image to train/val/test, stratified by source."""
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_source[record["source"]].append(i)

    rng = np.random.default_rng(seed)
    splits: list[str] = [""] * len(records)
    train_ratio, val_ratio, _ = SPLIT_RATIOS

    for indices in by_source.values():
        shuffled = [int(i) for i in rng.permutation(indices)]
        total = len(shuffled)
        n_train = int(round(total * train_ratio))
        n_val = int(round(total * val_ratio))
        for position, index in enumerate(shuffled):
            if position < n_train:
                splits[index] = "train"
            elif position < n_train + n_val:
                splits[index] = "val"
            else:
                splits[index] = "test"
    return splits


# --------------------------------------------------------------------------
# Invalid-file fixtures
# --------------------------------------------------------------------------


def write_invalid_samples(out_dir: Path, reference: Path | None) -> None:
    """Write files that are genuinely undecodable, for API rejection tests.

    Distinct from the ``corruption`` class, which is visibly damaged but decodes
    fine. These should be rejected with a 422 rather than analysed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    (out_dir / "empty.jpg").write_bytes(b"")
    (out_dir / "random_bytes.jpg").write_bytes(rng.integers(0, 256, 40_000, dtype=np.uint8).tobytes())
    (out_dir / "text_masquerading_as_png.png").write_bytes(b"this is not an image, it is prose.\n" * 200)
    (out_dir / "png_header_only.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)

    if reference is not None and reference.exists():
        data = reference.read_bytes()
        (out_dir / "truncated.jpg").write_bytes(data[: max(1, len(data) // 3)])

    print(f"Wrote invalid-file fixtures to {out_dir}")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _summarise(rows: list[dict]) -> None:
    total = len(rows)
    print(f"\nGenerated {total} images from {len({r['base_image_id'] for r in rows})} base images\n")

    split_counts = Counter(r["split"] for r in rows)
    print(f"{'split':<8} {'images':>8} {'share':>7}")
    for split in ("train", "val", "test"):
        count = split_counts.get(split, 0)
        print(f"{split:<8} {count:>8} {count / total:>6.1%}")

    clean = sum(r["is_clean"] for r in rows)
    print(f"\nclean (no issue): {clean} ({clean / total:.1%})")

    print(f"\n{'issue':<16} {'positives':>10} {'rate':>7}   {'low':>5} {'med':>5} {'high':>5}")
    for issue in ISSUE_TYPES:
        positives = [r for r in rows if r[f"{issue}_present"]]
        buckets = Counter(r[f"{issue}_severity_bucket"] for r in positives)
        print(
            f"{issue:<16} {len(positives):>10} {len(positives) / total:>6.1%}   "
            f"{buckets.get(1, 0):>5} {buckets.get(2, 0):>5} {buckets.get(3, 0):>5}"
        )

    label_counts = Counter(r["quality_label"] for r in rows)
    print(f"\n{'quality label':<16} {'images':>8} {'share':>7}")
    for label in ("EXCELLENT", "ACCEPTABLE", "POOR", "UNUSABLE"):
        count = label_counts.get(label, 0)
        print(f"{label:<16} {count:>8} {count / total:>6.1%}")

    scores = np.array([r["quality_score"] for r in rows])
    print(f"\nquality_score  mean {scores.mean():.1f}  median {np.median(scores):.1f}  "
          f"min {scores.min():.1f}  max {scores.max():.1f}")


def _write_manifest(rows: list[dict], out_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit(
            "pandas is required to write the manifest. Install the data extra:\n"
            "  pip install -e ./ml_pipeline[data]"
        )

    frame = pd.DataFrame(rows)
    csv_path = out_dir / "manifest.csv"
    frame.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    try:
        parquet_path = out_dir / "manifest.parquet"
        frame.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path}")
    except (ImportError, ValueError) as exc:
        print(f"Skipped parquet manifest ({exc}); CSV is sufficient for training.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, default=Path("dataset/raw/base_index.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--limit", type=int, default=None, help="use only the first N base images")
    parser.add_argument("--long-side", type=int, default=CANONICAL_LONG_SIDE)
    parser.add_argument("--jpeg-quality", type=int, default=GENERATED_JPEG_QUALITY)
    parser.add_argument("--clean-variants", type=int, default=2)
    parser.add_argument("--combo-variants", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20240828)
    parser.add_argument("--no-invalid-samples", action="store_true")
    args = parser.parse_args()

    if not args.index.exists():
        raise SystemExit(
            f"{args.index} not found. Run the downloader first:\n"
            "  python -m dataset.download_sources"
        )

    index = json.loads(args.index.read_text(encoding="utf-8"))
    records: list[dict] = index["images"]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("base index contains no images")

    config = GenConfig(
        out_dir=str(args.out_dir),
        long_side=args.long_side,
        jpeg_quality=args.jpeg_quality,
        clean_variants=args.clean_variants,
        combo_variants=args.combo_variants,
        seed=args.seed,
    )
    per_image = args.clean_variants + len(ISSUE_TYPES) + args.combo_variants
    print(
        f"Generating ~{len(records) * per_image} images "
        f"({per_image} variants x {len(records)} base images) into {args.out_dir}"
    )

    splits = assign_splits(records, args.seed)
    tasks = [(i, record, splits[i], config) for i, record in enumerate(records)]

    rows: list[dict] = []
    if args.workers <= 1:
        for task in tasks:
            rows.extend(generate_for_base(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(generate_for_base, task) for task in tasks]
            try:
                from tqdm import tqdm
                iterator = tqdm(as_completed(futures), total=len(futures), desc="base images")
            except ImportError:
                iterator = as_completed(futures)
            for future in iterator:
                rows.extend(future.result())

    if not rows:
        raise SystemExit("no images were generated — check that the base paths still exist")

    rows.sort(key=lambda r: r["image_path"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, args.out_dir)

    (args.out_dir / "generation_meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_base_images": len(records),
                "n_generated": len(rows),
                "split_ratios": list(SPLIT_RATIOS),
                "config": config.__dict__,
                "base_index": str(args.index),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _summarise(rows)

    if not args.no_invalid_samples:
        reference = Path(records[0]["path"])
        write_invalid_samples(Path("dataset/invalid_samples"), reference)

    print("\nNext: implement classical feature extraction (Phase 2)")


if __name__ == "__main__":
    main()
