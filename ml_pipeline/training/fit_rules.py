"""Refit the rule layer's ramp thresholds from the training split.

The defaults in ``focal_ml.fusion.rules`` are argued from the physical meaning
of each measurement. That makes them defensible but not optimal: the point at
which blur becomes visible genuinely depends on the image population. This
script replaces those numbers with percentiles measured on real data, and
reports what the change bought.

The ramp is placed so that:

    onset    = the value below which most clean images fall
    saturate = the value that clearly-affected images reach

Concretely, ``onset`` is a high percentile of the negative class and
``saturate`` a middling percentile of the *severe* positives. Fitting against
severe cases rather than all positives keeps a crowd of barely-visible
low-severity examples from dragging the saturation point down to where clean
images live.

Only the thresholds move — which features drive which issue, and how they
aggregate, stay as written. Those encode reasoning that should not be silently
rewritten by whatever correlates best in one corpus.

Usage (from ``ml_pipeline/``)::

    python -m training.fit_rules
    python -m training.fit_rules --out ../ml_pipeline/models/rules_v1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from focal_ml.constants import ISSUE_TYPES
from focal_ml.fusion.rules import Ramp, RampGroup, RuleConfig, Term, evaluate_rules

#: Percentile of the clean population used as the onset. Higher means fewer
#: false positives and later detection.
NEGATIVE_PERCENTILE = 90.0
#: Percentile of the severe-positive population used as the saturation point.
POSITIVE_PERCENTILE = 40.0
#: Severity above which a positive counts as "clearly affected" for fitting.
SEVERE_THRESHOLD = 0.55


def fit_ramp(ramp: Ramp, negatives: np.ndarray, positives: np.ndarray) -> Ramp:
    """Reposition one ramp, preserving its direction."""
    if negatives.size < 20 or positives.size < 10:
        return ramp

    ascending = ramp.saturate >= ramp.onset
    if ascending:
        onset = float(np.percentile(negatives, NEGATIVE_PERCENTILE))
        saturate = float(np.percentile(positives, 100.0 - POSITIVE_PERCENTILE))
    else:
        onset = float(np.percentile(negatives, 100.0 - NEGATIVE_PERCENTILE))
        saturate = float(np.percentile(positives, POSITIVE_PERCENTILE))

    # A degenerate fit means the feature does not separate the classes on this
    # corpus. Keeping the reasoned default is better than installing a ramp that
    # fires on everything or nothing.
    if ascending and saturate <= onset:
        return ramp
    if not ascending and saturate >= onset:
        return ramp

    return Ramp(feature=ramp.feature, onset=round(onset, 6), saturate=round(saturate, 6), weight=ramp.weight)


def fit_term(term: Term, frame, present: np.ndarray, severe: np.ndarray) -> Term:
    """Refit a term, descending into conjunction groups.

    Each member of a group is fitted independently against the same positive
    and negative populations. The group's structure — which conditions must
    co-occur — is reasoning that stays fixed; only where each condition sits is
    learned.
    """
    if isinstance(term, RampGroup):
        return RampGroup(
            ramps=tuple(
                fit_ramp(ramp, frame[ramp.feature].to_numpy(dtype=np.float64)[~present],
                         frame[ramp.feature].to_numpy(dtype=np.float64)[severe])
                for ramp in term.ramps
            ),
            weight=term.weight,
            label=term.label,
        )
    values = frame[term.feature].to_numpy(dtype=np.float64)
    return fit_ramp(term, values[~present], values[severe])


def _term_summary(term: Term) -> list[tuple[str, float, float]]:
    """Flatten a term into (feature, onset, saturate) rows for reporting."""
    if isinstance(term, RampGroup):
        return [(ramp.feature, ramp.onset, ramp.saturate) for ramp in term.ramps]
    return [(term.feature, term.onset, term.saturate)]


def score_config(config: RuleConfig, frame) -> dict[str, dict[str, float]]:
    """Per-issue precision / recall / F1 of the rule layer alone."""
    results: dict[str, dict[str, float]] = {}
    feature_dicts = frame.to_dict("records")
    outcomes = [evaluate_rules(row, config) for row in feature_dicts]

    for issue in ISSUE_TYPES:
        threshold = config.rules[issue].report_threshold
        predicted = np.array([outcome[issue].confidence >= threshold for outcome in outcomes])
        actual = frame[f"{issue}_present"].to_numpy().astype(bool)

        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[issue] = {"precision": precision, "recall": recall, "f1": f1, "support": int(actual.sum())}

    return results


def _print_scores(title: str, scores: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"  {'issue':<16} {'precision':>9} {'recall':>7} {'F1':>7} {'support':>8}")
    for issue, row in scores.items():
        print(f"  {issue:<16} {row['precision']:>9.3f} {row['recall']:>7.3f} {row['f1']:>7.3f} {row['support']:>8}")
    mean_f1 = float(np.mean([row["f1"] for row in scores.values()]))
    print(f"  {'mean F1':<16} {mean_f1:>26.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--out", type=Path, default=Path("models/rules_v1.json"))
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required: pip install -e '.[data]'")

    manifest_path = args.generated_dir / "manifest.csv"
    features_path = args.generated_dir / "features.parquet"
    if not features_path.exists():
        features_path = args.generated_dir / "features.csv"
    if not (manifest_path.exists() and features_path.exists()):
        raise SystemExit(
            "Need both manifest and features. Run:\n"
            "  python -m dataset.generate_synthetic\n"
            "  python -m dataset.extract_features"
        )

    manifest = pd.read_csv(manifest_path)
    features = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    merged = manifest.merge(features, on="image_path")

    train = merged[merged["split"] == "train"]
    validation = merged[merged["split"] == "val"]
    print(f"Fitting on {len(train)} training images, validating on {len(validation)}")

    defaults = RuleConfig()
    fitted = RuleConfig(version="rules_v1_fitted")

    for issue in ISSUE_TYPES:
        present = train[f"{issue}_present"].to_numpy().astype(bool)
        severe = present & (train[f"{issue}_severity_score"].to_numpy() >= SEVERE_THRESHOLD)

        fitted.rules[issue].ramps = [
            fit_term(term, train, present, severe) for term in fitted.rules[issue].ramps
        ]

    _print_scores("Reasoned defaults (validation split)", score_config(defaults, validation))
    _print_scores("Fitted thresholds (validation split)", score_config(fitted, validation))

    print("\nThreshold changes:")
    for issue in ISSUE_TYPES:
        for old_term, new_term in zip(defaults.rules[issue].ramps, fitted.rules[issue].ramps):
            for (feature, old_on, old_sat), (_, new_on, new_sat) in zip(
                _term_summary(old_term), _term_summary(new_term)
            ):
                if (old_on, old_sat) != (new_on, new_sat):
                    print(f"  {issue:<14} {feature:<28} "
                          f"[{old_on:.4g} -> {old_sat:.4g}]  becomes  "
                          f"[{new_on:.4g} -> {new_sat:.4g}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_json(args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
