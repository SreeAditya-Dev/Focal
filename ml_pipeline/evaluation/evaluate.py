"""Evaluate the trained system on the held-out test split.

Produces per-issue detection metrics, severity regression error, quality-score
agreement, confusion matrices, calibration curves, an ablation comparison and a
set of concrete failure cases.

Three things here are deliberate:

**The whole pipeline is evaluated, not the network.** Every number comes from
the same fusion the API serves, so what is measured is the product's behaviour
rather than one component's. A model that scores well while the rules override
it would look good under a network-only evaluation and bad in use.

**Rules-only is always evaluated as a baseline.** Without it "the CNN gets 0.9
AUC" is unanchored — the interesting question is what the learned component adds
over measurements anyone could have written, and on which issues.

**Real degradation is tested separately from synthetic.** The entire corpus is
synthetically degraded, so test-split performance measures how well the model
learned *this generator*. The images rejected during screening are real
photographs that genuinely failed a sharpness or exposure check, which makes
them a small but honest domain-gap probe.

Usage (from ``ml_pipeline/``)::

    python -m evaluation.evaluate
    python -m evaluation.evaluate --ablation      # compare all model variants
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from focal_ml.constants import (
    ISSUE_TYPES,
    QUALITY_LABELS,
    compute_quality_score,
    quality_label,
)
from focal_ml.fusion.rules import RuleConfig, evaluate_rules
from focal_ml.fusion.scorer import REPORT_THRESHOLD, fuse
from focal_ml.model.architecture import FocalNet, ModelConfig
from focal_ml.model.calibration import Calibration, expected_calibration_error
from evaluation.metrics import (
    best_f1_threshold,
    binary_metrics,
    confusion_matrix,
    label_accuracy,
    macro_summary,
    reliability_curve,
    score_regression_metrics,
    severity_metrics,
)
from training.dataset import load_split


def load_model(path: Path, device: torch.device) -> tuple[FocalNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    config.pretrained = False
    model = FocalNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def run_model(model: FocalNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Presence logits and severity predictions over a split."""
    presence, severity = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {}
            if model.config.use_image:
                inputs["image"] = batch["image"].to(device)
            if model.config.use_features:
                inputs["features"] = batch["features"].to(device)
            output = model(**inputs)
            presence.append(output["presence_logits"].cpu().numpy())
            severity.append(torch.sigmoid(output["severity_logits"]).cpu().numpy())
    return np.concatenate(presence), np.concatenate(severity)


def fused_predictions(
    feature_rows: list[dict],
    cnn_presence: np.ndarray | None,
    cnn_severity: np.ndarray | None,
    rules: RuleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run the production fusion over a split.

    Returns fused confidences, fused severities, quality scores and labels.
    """
    confidences = np.zeros((len(feature_rows), len(ISSUE_TYPES)), dtype=np.float32)
    severities = np.zeros_like(confidences)
    scores = np.zeros(len(feature_rows), dtype=np.float32)
    labels: list[str] = []

    for row_index, features in enumerate(feature_rows):
        outcomes = evaluate_rules(features, rules)
        presence = (
            {issue: float(cnn_presence[row_index, i]) for i, issue in enumerate(ISSUE_TYPES)}
            if cnn_presence is not None else None
        )
        severity = (
            {issue: float(cnn_severity[row_index, i]) for i, issue in enumerate(ISSUE_TYPES)}
            if cnn_severity is not None else None
        )
        result = fuse(outcomes, presence, severity)

        for i, issue in enumerate(ISSUE_TYPES):
            confidences[row_index, i] = result.all_confidences[issue]
            severities[row_index, i] = result.all_severities[issue]
        scores[row_index] = result.quality_score
        labels.append(result.quality_label)

    return confidences, severities, scores, labels


def evaluate_variant(
    name: str,
    feature_rows: list[dict],
    manifest,
    cnn_presence: np.ndarray | None,
    cnn_severity: np.ndarray | None,
    rules: RuleConfig,
) -> dict:
    """Full metric set for one configuration of the pipeline."""
    true_presence = manifest[[f"{i}_present" for i in ISSUE_TYPES]].to_numpy(np.float32)
    true_severity = manifest[[f"{i}_severity_score" for i in ISSUE_TYPES]].to_numpy(np.float32)
    true_scores = manifest["quality_score"].to_numpy(np.float64)
    true_labels = manifest["quality_label"].tolist()

    confidences, severities, scores, labels = fused_predictions(
        feature_rows, cnn_presence, cnn_severity, rules
    )

    per_issue, severity_rows = [], []
    for index, issue in enumerate(ISSUE_TYPES):
        metrics = binary_metrics(
            true_presence[:, index], confidences[:, index], issue, threshold=REPORT_THRESHOLD
        )
        threshold, f1 = best_f1_threshold(true_presence[:, index], confidences[:, index])
        row = metrics.to_dict()
        row["best_f1"] = round(f1, 4)
        row["best_f1_threshold"] = round(threshold, 4)
        per_issue.append(row)

        severity_rows.append(asdict(severity_metrics(
            true_severity[:, index], severities[:, index], true_presence[:, index], issue
        )))

    return {
        "variant": name,
        "n_images": len(manifest),
        "per_issue": per_issue,
        "summary": macro_summary([
            binary_metrics(true_presence[:, i], confidences[:, i], issue, REPORT_THRESHOLD)
            for i, issue in enumerate(ISSUE_TYPES)
        ]),
        "severity": severity_rows,
        "quality_score": score_regression_metrics(true_scores, scores.astype(np.float64)),
        "quality_label": label_accuracy(true_labels, labels),
        "label_confusion": confusion_matrix(
            np.array(true_labels), np.array(labels), list(QUALITY_LABELS)
        ).tolist(),
        "_raw": {"confidences": confidences, "scores": scores, "labels": labels},
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_variant(result: dict) -> None:
    print(f"\n{'=' * 78}\n{result['variant']}  ({result['n_images']} test images)\n{'=' * 78}")
    print(f"{'issue':<15} {'prec':>6} {'recall':>7} {'F1':>6} {'ROC':>6} {'PR':>6} "
          f"{'bestF1':>7} {'@thr':>6} {'n':>6}")
    for row in result["per_issue"]:
        print(f"{row['issue']:<15} {row['precision']:>6.3f} {row['recall']:>7.3f} {row['f1']:>6.3f} "
              f"{row['roc_auc']:>6.3f} {row['pr_auc']:>6.3f} {row['best_f1']:>7.3f} "
              f"{row['best_f1_threshold']:>6.2f} {row['support']:>6}")

    summary = result["summary"]
    print(f"{'MACRO':<15} {summary['macro_precision']:>6.3f} {summary['macro_recall']:>7.3f} "
          f"{summary['macro_f1']:>6.3f} {summary['macro_roc_auc']:>6.3f} {summary['macro_pr_auc']:>6.3f}")

    print(f"\n{'severity (positives only)':<28} {'MAE':>7} {'RMSE':>7} {'bucket':>8} {'+-1':>7}")
    for row in result["severity"]:
        print(f"  {row['issue']:<26} {row['mae']:>7.3f} {row['rmse']:>7.3f} "
              f"{row['bucket_accuracy']:>8.3f} {row['bucket_accuracy_within_one']:>7.3f}")

    score = result["quality_score"]
    print(f"\nquality_score   MAE {score['mae']:.2f}   RMSE {score['rmse']:.2f}   "
          f"bias {score['bias']:+.2f}   Pearson {score['pearson_r']:.3f}   Spearman {score['spearman_rho']:.3f}")
    print(f"quality_label   exact {result['quality_label']['exact']:.3f}   "
          f"within one band {result['quality_label']['within_one_band']:.3f}")

    print(f"\nlabel confusion (rows true, cols predicted): {', '.join(QUALITY_LABELS)}")
    for name, row in zip(QUALITY_LABELS, result["label_confusion"]):
        print(f"  {name:<12} " + " ".join(f"{v:>6}" for v in row))


def write_plots(
    result: dict, true_presence: np.ndarray, calibration: Calibration | None, out_dir: Path
) -> None:
    """Reliability diagram and per-issue PR curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    confidences = result["_raw"]["confidences"]

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for index, issue in enumerate(ISSUE_TYPES):
        axis = axes[index // 3][index % 3]
        confidence, accuracy, counts = reliability_curve(confidences[:, index], true_presence[:, index])
        keep = counts > 0
        axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        axis.plot(confidence[keep], accuracy[keep], "o-", label="observed")
        ece = expected_calibration_error(confidences[:, index], true_presence[:, index])
        axis.set_title(f"{issue}  (ECE {ece:.3f})")
        axis.set_xlabel("predicted confidence")
        axis.set_ylabel("observed frequency")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)

    figure.suptitle(f"Calibration — {result['variant']}")
    figure.tight_layout()
    figure.savefig(out_dir / "reliability.png", dpi=110)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    from sklearn.metrics import precision_recall_curve

    for index, issue in enumerate(ISSUE_TYPES):
        labels = true_presence[:, index].astype(int)
        if labels.min() == labels.max():
            continue
        precision, recall, _ = precision_recall_curve(labels, confidences[:, index])
        axis.plot(recall, precision, label=f"{issue} (AP {result['per_issue'][index]['pr_auc']:.3f})")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(f"Precision-recall — {result['variant']}")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "precision_recall.png", dpi=110)
    plt.close(figure)

    print(f"Wrote plots to {out_dir}")


def collect_failure_cases(
    result: dict, manifest, out_dir: Path, generated_dir: Path, top: int = 12
) -> list[dict]:
    """The worst quality-score errors, saved with their images for inspection.

    Aggregate metrics say how often the system is wrong; these say what being
    wrong looks like, which is what actually guides the next iteration.
    """
    import shutil

    predicted = result["_raw"]["scores"]
    actual = manifest["quality_score"].to_numpy(np.float64)
    errors = predicted - actual
    order = np.argsort(-np.abs(errors))[:top]

    failures_dir = out_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for rank, index in enumerate(order):
        row = manifest.iloc[int(index)]
        present = [issue for issue in ISSUE_TYPES if row[f"{issue}_present"]]
        detected = [
            issue for i, issue in enumerate(ISSUE_TYPES)
            if result["_raw"]["confidences"][index, i] >= REPORT_THRESHOLD
        ]
        record = {
            "rank": rank + 1,
            "image_path": row["image_path"],
            "variant": row["variant"],
            "true_score": float(actual[index]),
            "predicted_score": float(predicted[index]),
            "error": round(float(errors[index]), 2),
            "true_issues": present,
            "detected_issues": detected,
            "missed": [i for i in present if i not in detected],
            "spurious": [i for i in detected if i not in present],
        }
        rows.append(record)

        source = generated_dir / row["image_path"]
        if source.exists():
            shutil.copy(source, failures_dir / f"{rank + 1:02d}_{Path(row['image_path']).name}")

    (out_dir / "failure_cases.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWorst {len(rows)} quality-score errors -> {failures_dir}")
    print(f"  {'err':>7}  {'true':>5} {'pred':>5}  missed / spurious")
    for record in rows[:8]:
        print(f"  {record['error']:>+7.1f}  {record['true_score']:>5.1f} {record['predicted_score']:>5.1f}  "
              f"{','.join(record['missed']) or '-'} / {','.join(record['spurious']) or '-'}")
    return rows


def evaluate_real_degradation(predictor, raw_dir: Path, limit: int = 150) -> dict | None:
    """Probe the synthetic-to-real gap using images rejected during screening.

    Every training image was degraded by our own generator, so test-split
    numbers measure how well the model learned *that generator*. The photographs
    rejected by the base-image screen are real ones that genuinely failed a
    sharpness, exposure or contrast check — weakly labelled, but authentically
    degraded, which is exactly the axis synthetic data cannot cover.

    This is a sanity probe rather than a benchmark: the labels say only "this
    failed some check", not which issue a human would name.
    """
    import cv2

    from dataset.download_sources import (
        BRIGHTNESS_RANGE, MIN_CONTRAST, MIN_SHARPNESS, SCREEN_LONG_SIDE,
    )
    from focal_ml.utils import imread_bgr, resize_long_side, to_gray

    index_path = raw_dir / "base_index.json"
    if not index_path.exists():
        return None
    accepted = {record["path"] for record in json.loads(index_path.read_text())["images"]}

    candidates = []
    for source in ("coco", "div2k"):
        directory = raw_dir / source
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if str(path.resolve()) in accepted:
                continue
            candidates.append(path)
            if len(candidates) >= limit * 4:
                break

    tallies = {"soft": [], "dark": [], "bright": [], "flat": []}
    for path in candidates:
        image = imread_bgr(path)
        if image is None:
            continue
        gray = to_gray(resize_long_side(image, SCREEN_LONG_SIDE))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness, contrast = float(gray.mean()), float(gray.std())

        if sharpness < MIN_SHARPNESS:
            bucket = "soft"
        elif brightness < BRIGHTNESS_RANGE[0]:
            bucket = "dark"
        elif brightness > BRIGHTNESS_RANGE[1]:
            bucket = "bright"
        elif contrast < MIN_CONTRAST:
            bucket = "flat"
        else:
            continue

        if len(tallies[bucket]) >= limit // 4:
            continue
        result = predictor.analyse(image, include_heatmap=False)
        tallies[bucket].append({
            "score": result.quality_score,
            "issues": [issue.type for issue in result.issues],
        })

    expected = {"soft": "blur", "dark": "underexposure", "bright": "overexposure", "flat": None}
    report = {}
    for bucket, entries in tallies.items():
        if not entries:
            continue
        target = expected[bucket]
        detected = (
            sum(1 for e in entries if target in e["issues"]) / len(entries) if target else None
        )
        report[bucket] = {
            "n": len(entries),
            "expected_issue": target,
            "detection_rate": round(detected, 3) if detected is not None else None,
            "mean_score": round(float(np.mean([e["score"] for e in entries])), 1),
            "flagged_any_issue": round(
                float(np.mean([1.0 if e["issues"] else 0.0 for e in entries])), 3
            ),
        }
    return report


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/focal_cnn_v1.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("models/calibration_v1.json"))
    parser.add_argument("--rules", type=Path, default=Path("models/rules_v1.json"))
    parser.add_argument("--generated-dir", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument("--ablation", action="store_true", help="also evaluate the other variants")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-real-check", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rules = RuleConfig.from_json(args.rules) if args.rules.exists() else RuleConfig()
    calibration = (
        Calibration.from_json(args.calibration) if args.calibration.exists() else Calibration.identity()
    )
    print(f"rules: {rules.version}   calibration: {calibration.version}   device: {device}")

    manifest = pd.read_csv(args.generated_dir / "manifest.csv")
    manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    features_path = args.generated_dir / "features.parquet"
    if not features_path.exists():
        features_path = args.generated_dir / "features.csv"
    features = (
        pd.read_parquet(features_path) if features_path.suffix == ".parquet"
        else pd.read_csv(features_path)
    )
    # Aligned by path, never by merge order: a silent permutation would score
    # every image against another image's measurements and still look plausible.
    aligned = features.set_index("image_path").loc[manifest["image_path"]]
    feature_rows = aligned.to_dict("records")

    true_presence = manifest[[f"{i}_present" for i in ISSUE_TYPES]].to_numpy(np.float32)
    results = []

    # Rules-only: the baseline that makes every other number interpretable.
    results.append(evaluate_variant("rules only (no CNN)", feature_rows, manifest, None, None, rules))

    variants = [("hybrid (image + features)", args.model)]
    if args.ablation:
        variants += [
            ("CNN image only", args.model.parent / "focal_cnn_image_only.pt"),
            ("CNN features only", args.model.parent / "focal_cnn_features_only.pt"),
        ]

    primary = None
    for name, path in variants:
        if not path.exists():
            print(f"skipping '{name}': {path} not found")
            continue

        model, _ = load_model(path, device)
        test_set = load_split(
            args.generated_dir, "test",
            with_features=model.config.use_features, load_image=model.config.use_image,
        )
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        logits, severities = run_model(model, loader, device)
        probabilities = calibration.apply(logits)

        result = evaluate_variant(name, feature_rows, manifest, probabilities, severities, rules)
        results.append(result)
        if primary is None:
            primary = result

    for result in results:
        print_variant(result)

    if len(results) > 1:
        print(f"\n{'=' * 78}\nVARIANT COMPARISON\n{'=' * 78}")
        print(f"{'variant':<28} {'macroF1':>8} {'macroROC':>9} {'macroPR':>8} "
              f"{'scoreMAE':>9} {'labelAcc':>9}")
        for result in results:
            print(f"{result['variant']:<28} {result['summary']['macro_f1']:>8.3f} "
                  f"{result['summary']['macro_roc_auc']:>9.3f} {result['summary']['macro_pr_auc']:>8.3f} "
                  f"{result['quality_score']['mae']:>9.2f} {result['quality_label']['exact']:>9.3f}")

    report = primary or results[0]
    write_plots(report, true_presence, calibration, args.out_dir)
    failures = collect_failure_cases(report, manifest, args.out_dir, args.generated_dir)

    real_check = None
    if not args.no_real_check:
        from focal_ml.inference import FocalPredictor

        predictor = FocalPredictor(
            model_path=args.model if args.model.exists() else None,
            calibration_path=args.calibration if args.calibration.exists() else None,
            rules_path=args.rules if args.rules.exists() else None,
        )
        real_check = evaluate_real_degradation(predictor, args.raw_dir)
        if real_check:
            print(f"\n{'=' * 78}\nREAL (NON-SYNTHETIC) DEGRADATION PROBE\n{'=' * 78}")
            print(f"{'rejected for':<12} {'n':>4} {'expected':<15} {'detected':>9} "
                  f"{'anyIssue':>9} {'meanScore':>10}")
            for bucket, row in real_check.items():
                rate = f"{row['detection_rate']:.3f}" if row["detection_rate"] is not None else "n/a"
                print(f"{bucket:<12} {row['n']:>4} {str(row['expected_issue']):<15} {rate:>9} "
                      f"{row['flagged_any_issue']:>9.3f} {row['mean_score']:>10.1f}")

    payload = {
        "variants": [{k: v for k, v in r.items() if k != "_raw"} for r in results],
        "failure_cases": failures,
        "real_degradation_probe": real_check,
        "rules_version": rules.version,
        "calibration_version": calibration.version,
    }
    (args.out_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
