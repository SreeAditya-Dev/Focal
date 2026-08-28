"""Fit temperature scaling and the fusion weights on the validation split.

Both are fitted on validation rather than training data, and for the same
reason: the model is overconfident *because* it fits the training set, so
temperatures fitted there would be near 1.0 and correct nothing. The same
applies to the rule/CNN blend — measured on training data the CNN looks better
than it is, and the blend would lean on it too heavily.

Neither step touches the network's weights. Temperature scaling is monotonic, so
it cannot change any ranking or AUC; it only moves probabilities onto a scale
where 0.7 means 70%. The fusion weights change how two existing opinions are
combined, not either opinion.

Usage (from ``ml_pipeline/``)::

    python -m training.calibrate --model models/focal_cnn_v1.pt
    python -m training.calibrate --model models/focal_cnn_v1.pt --tune-fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES
from focal_ml.fusion.rules import RuleConfig, evaluate_rules
from focal_ml.fusion.scorer import RULE_WEIGHT, fuse
from focal_ml.model.architecture import FocalNet, ModelConfig
from focal_ml.model.calibration import expected_calibration_error, fit_calibration
from training.dataset import load_split

#: Candidate rule weights swept when --tune-fusion is given.
WEIGHT_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0)


def collect_logits(
    model: FocalNet, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a split, returning raw logits and labels."""
    model.eval()
    logits, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs = {}
            if model.config.use_image:
                inputs["image"] = batch["image"].to(device)
            if model.config.use_features:
                inputs["features"] = batch["features"].to(device)
            output = model(**inputs)
            logits.append(output["presence_logits"].cpu().numpy())
            labels.append(batch["presence"].numpy())

    return np.concatenate(logits), np.concatenate(labels)


def _f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    true_positive = float((predicted & actual).sum())
    if true_positive == 0:
        return 0.0
    precision = true_positive / float(predicted.sum())
    recall = true_positive / float(actual.sum())
    return 2 * precision * recall / (precision + recall)


def tune_fusion_weights(
    rule_outcomes: list[dict],
    cnn_presence: list[dict],
    cnn_severity: list[dict],
    labels: np.ndarray,
) -> dict[str, float]:
    """Sweep each issue's rule weight for the best validation F1.

    Swept per issue rather than as one global constant, because how far the
    rules can be trusted differs sharply between issues — a single alpha would
    either waste the rules' reliability on exposure or over-trust them on
    defects.
    """
    tuned: dict[str, float] = {}

    for index, issue in enumerate(ISSUE_TYPES):
        actual = labels[:, index] > 0.5
        best_weight, best_score = RULE_WEIGHT.get(issue, 0.4), -1.0

        for weight in WEIGHT_GRID:
            confidences = np.array([
                weight * rules[issue].confidence + (1.0 - weight) * presence[issue]
                for rules, presence in zip(rule_outcomes, cnn_presence)
            ])
            score = _f1(confidences >= 0.35, actual)
            if score > best_score:
                best_score, best_weight = score, weight

        tuned[issue] = best_weight
        print(f"  {issue:<16} weight {best_weight:.2f}  F1 {best_score:.3f}")

    return tuned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/focal_cnn_v1.pt"))
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--out", type=Path, default=Path("models/calibration_v1.json"))
    parser.add_argument("--rules", type=Path, default=None, help="fitted rules JSON, if any")
    parser.add_argument("--tune-fusion", action="store_true", help="also sweep the rule/CNN blend")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Train first: python -m training.train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)

    config = ModelConfig(**checkpoint["model_config"])
    config.pretrained = False
    model = FocalNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    val_set = load_split(
        args.generated_dir, "val",
        with_features=config.use_features, load_image=config.use_image,
    )
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print(f"Calibrating on {len(val_set)} validation images")

    logits, labels = collect_logits(model, loader, device)
    calibration = fit_calibration(logits, labels)

    print(f"\n{'issue':<16} {'T':>6} {'ECE before':>11} {'ECE after':>10} {'change':>9}")
    for issue in ISSUE_TYPES:
        before = calibration.ece_before[issue]
        after = calibration.ece_after[issue]
        arrow = "better" if after < before else "worse" if after > before else "same"
        print(f"{issue:<16} {calibration.temperatures[issue]:>6.3f} {before:>11.4f} {after:>10.4f} {arrow:>9}")

    mean_before = float(np.mean(list(calibration.ece_before.values())))
    mean_after = float(np.mean(list(calibration.ece_after.values())))
    print(f"{'mean':<16} {'':>6} {mean_before:>11.4f} {mean_after:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_json(args.out)
    print(f"\nWrote {args.out}")

    if args.tune_fusion:
        print("\nTuning the rule/CNN blend per issue:")
        rule_config = RuleConfig.from_json(args.rules) if args.rules and args.rules.exists() else RuleConfig()

        import pandas as pd

        features_path = args.generated_dir / "features.parquet"
        if not features_path.exists():
            features_path = args.generated_dir / "features.csv"
        frame = (
            pd.read_parquet(features_path) if features_path.suffix == ".parquet"
            else pd.read_csv(features_path)
        )
        # Aligned by path against the dataset's own ordering rather than by
        # merging and trusting the result to come back in the same order. The
        # model outputs are in DataLoader order; a merge that silently permuted
        # the rows would pair each image's model prediction with a different
        # image's rule outcome, and every number downstream would still look
        # entirely plausible.
        indexed = frame.set_index("image_path")
        missing = set(val_set.paths) - set(indexed.index)
        if missing:
            raise SystemExit(f"{len(missing)} validation images have no extracted features")
        aligned = indexed.loc[val_set.paths]

        rule_outcomes = [evaluate_rules(row, rule_config) for row in aligned.to_dict("records")]

        probabilities = calibration.apply(logits)
        severity_batches = []
        with torch.no_grad():
            for batch in loader:
                inputs = {}
                if config.use_image:
                    inputs["image"] = batch["image"].to(device)
                if config.use_features:
                    inputs["features"] = batch["features"].to(device)
                severity_batches.append(torch.sigmoid(model(**inputs)["severity_logits"]).cpu().numpy())
        severities = np.concatenate(severity_batches)

        cnn_presence = [
            {issue: float(row[i]) for i, issue in enumerate(ISSUE_TYPES)} for row in probabilities
        ]
        cnn_severity = [
            {issue: float(row[i]) for i, issue in enumerate(ISSUE_TYPES)} for row in severities
        ]

        assert len(rule_outcomes) == len(cnn_presence) == len(labels)

        tuned = tune_fusion_weights(rule_outcomes, cnn_presence, cnn_severity, labels)
        weights_path = args.out.parent / "fusion_weights.json"
        weights_path.write_text(json.dumps({"rule_weight": tuned}, indent=2), encoding="utf-8")
        print(f"\nWrote {weights_path}")
        print("Defaults for comparison: " + ", ".join(f"{k}={v}" for k, v in RULE_WEIGHT.items()))


if __name__ == "__main__":
    main()
