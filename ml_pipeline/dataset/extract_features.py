"""Compute classical features for every image in a generated manifest.

Produces `features.parquet` alongside the manifest, joined on `image_path`.
Three later stages consume it: rule-threshold fitting, the features-only
baseline model, and the hybrid model's auxiliary input.

Usage (from ``ml_pipeline/``)::

    python -m dataset.extract_features
    python -m dataset.extract_features --report    # print discriminative power
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES, extract_from_path


def _init_worker() -> None:
    """Confine each worker to a single thread.

    OpenCV and the BLAS backends each start a thread pool sized to the machine's
    core count, and every thread allocates its own scratch buffers. Multiplied
    across worker processes that is dozens of pools on an eight-core box, which
    exhausted 16 GB partway through a full extraction run.

    When the work is already parallel across processes, intra-operation
    threading buys nothing — the cores are saturated either way — so restricting
    it costs no throughput and removes the memory multiplier.
    """
    import cv2

    cv2.setNumThreads(1)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def _extract_one(task: tuple[str, str]) -> dict | None:
    image_path, root = task
    try:
        features = extract_from_path(Path(root) / image_path)
    except (MemoryError, ValueError, OSError):
        # One unreadable or pathological image must not abort a run that is
        # twenty minutes deep. Missing rows are reported at the end.
        return None
    if features is None:
        return None
    return {"image_path": image_path, **features}


def separation_report(frame, manifest) -> None:
    """Rank features by how well each separates present from absent, per issue.

    Uses AUC — the probability that a random positive scores above a random
    negative — because it is threshold-free and invariant to the wildly
    different scales these features live on. A value near 0.5 means the feature
    carries no information about that issue; near 0 means it is informative but
    inversely related.
    """
    merged = manifest.merge(frame, on="image_path")

    for issue in ISSUE_TYPES:
        labels = merged[f"{issue}_present"].to_numpy()
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        scores = []
        for name in FEATURE_NAMES:
            values = merged[name].to_numpy(dtype=np.float64)
            if np.allclose(values, values[0]):
                continue
            # AUC via the rank-sum identity, which avoids a sklearn dependency
            # and is exact including ties.
            order = values.argsort()
            ranks = np.empty(len(values), dtype=np.float64)
            ranks[order] = np.arange(1, len(values) + 1)
            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            scores.append((abs(auc - 0.5) * 2, auc, name))

        scores.sort(reverse=True)
        print(f"\n{issue}  ({int(labels.sum())} positives / {len(labels)})")
        print(f"  {'feature':<32} {'AUC':>6}  {'separation':>10}")
        for strength, auc, name in scores[:6]:
            direction = "higher" if auc > 0.5 else "lower"
            print(f"  {name:<32} {auc:>6.3f}  {strength:>9.2f}  ({direction} when present)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="reuse a checkpoint from an interrupted run (default on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="rows between checkpoint writes")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N manifest rows")
    parser.add_argument("--report", action="store_true", help="print per-issue feature AUCs")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required: pip install -e '.[data]'")

    manifest_path = args.generated_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found. Run: python -m dataset.generate_synthetic")

    manifest = pd.read_csv(manifest_path)
    if args.limit:
        manifest = manifest.head(args.limit)
    root = str(args.generated_dir)
    wanted = list(manifest["image_path"])

    # Extraction over the full corpus takes tens of minutes. Results are
    # checkpointed so an interruption costs the last few hundred images rather
    # than the whole run.
    partial_path = args.generated_dir / "features.partial.csv"
    done: dict[str, dict] = {}
    if args.resume and partial_path.exists():
        existing = pd.read_csv(partial_path)
        done = {row["image_path"]: row for row in existing.to_dict("records")}
        print(f"resuming: {len(done)} images already extracted")

    tasks = [(path, root) for path in wanted if path not in done]
    print(f"Extracting {len(FEATURE_NAMES)} features from {len(tasks)} images "
          f"({args.workers} workers, single-threaded each)...")

    rows: list[dict] = list(done.values())
    flushed = len(rows)

    def flush() -> None:
        nonlocal flushed
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        flushed = len(rows)

    if tasks:
        if args.workers <= 1:
            _init_worker()
            stream = (_extract_one(task) for task in tasks)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
            # A small chunk keeps results streaming back steadily, so the
            # checkpoint stays close to the work actually completed.
            stream = pool.map(_extract_one, tasks, chunksize=16)

        try:
            from tqdm import tqdm
            stream = tqdm(stream, total=len(tasks), desc="features")
        except ImportError:
            pass

        try:
            for result in stream:
                if result is not None:
                    rows.append(result)
                if len(rows) - flushed >= args.checkpoint_every:
                    flush()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            flush()

    frame = pd.DataFrame(rows)
    missing = len(wanted) - len(frame)
    if missing:
        share = missing / len(wanted)
        # A handful of unreadable images is tolerable; most of them failing is a
        # structural problem — wrong directory, moved corpus, exhausted memory —
        # and writing a mostly-empty feature table would let it pass silently
        # into training, where it would be far harder to diagnose.
        if share > 0.05:
            raise SystemExit(
                f"{missing}/{len(wanted)} images ({share:.0%}) produced no features.\n"
                f"Check that {args.generated_dir} contains the generated images, "
                "not just the manifest."
            )
        print(f"WARNING: {missing} images produced no features and were skipped")

    out_path = args.generated_dir / "features.parquet"
    try:
        frame.to_parquet(out_path, index=False)
    except (ImportError, ValueError):
        out_path = args.generated_dir / "features.csv"
        frame.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(frame)} rows x {len(FEATURE_NAMES)} features)")
    partial_path.unlink(missing_ok=True)

    if args.report:
        separation_report(frame, manifest)


if __name__ == "__main__":
    main()
