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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES, extract_from_path


def _extract_one(task: tuple[str, str]) -> dict | None:
    image_path, root = task
    features = extract_from_path(Path(root) / image_path)
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
    root = str(args.generated_dir)
    tasks = [(path, root) for path in manifest["image_path"]]
    print(f"Extracting {len(FEATURE_NAMES)} features from {len(tasks)} images...")

    if args.workers <= 1:
        rows = [_extract_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            chunk = max(1, len(tasks) // (args.workers * 8))
            try:
                from tqdm import tqdm
                rows = list(tqdm(pool.map(_extract_one, tasks, chunksize=chunk), total=len(tasks)))
            except ImportError:
                rows = list(pool.map(_extract_one, tasks, chunksize=chunk))

    rows = [row for row in rows if row is not None]
    frame = pd.DataFrame(rows)

    out_path = args.generated_dir / "features.parquet"
    try:
        frame.to_parquet(out_path, index=False)
    except (ImportError, ValueError):
        out_path = args.generated_dir / "features.csv"
        frame.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(frame)} rows x {len(FEATURE_NAMES)} features)")

    if args.report:
        separation_report(frame, manifest)


if __name__ == "__main__":
    main()
