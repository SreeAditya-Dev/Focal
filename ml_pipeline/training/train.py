"""Two-phase transfer-learning run for FocalNet.

Phase A freezes the backbone and trains only the heads. Phase B reopens the last
few backbone blocks at a much lower learning rate. Doing it in one phase instead
wastes the pretrained weights: a randomly initialised head produces large
gradients for the first few hundred steps, and letting those reach the backbone
destroys the ImageNet filters before the head has learned anything worth
propagating.

Usage (from ``ml_pipeline/``)::

    python -m training.train                       # full hybrid run
    python -m training.train --smoke               # tiny run, checks wiring
    python -m training.train --ablation image      # image-only, for Phase 5
    python -m training.train --ablation features   # features-only
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from focal_ml.constants import ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES
from focal_ml.model.architecture import FocalNet, ModelConfig
from training.dataset import feature_statistics, load_split, positive_weights


@dataclass
class TrainConfig:
    generated_dir: str = "dataset/generated"
    out_dir: str = "models"
    model_name: str = "focal_cnn_v1"

    head_epochs: int = 10
    finetune_epochs: int = 15
    batch_size: int = 32
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5
    finetune_head_lr: float = 1e-4
    weight_decay: float = 1e-4
    unfreeze_blocks: int = 2
    severity_weight: float = 1.5
    early_stopping_patience: int = 3
    num_workers: int = 4
    seed: int = 1234
    #: Cached frozen-backbone embeddings for Phase A, if built.
    embeddings_dir: str | None = None
    #: Cap on training images used in Phase B. Fine-tuning the last two blocks
    #: is a refinement of a head already trained on the full corpus, so it can
    #: run on a subsample when full-corpus backprop is not affordable.
    finetune_subset: int | None = None


class FocalLoss(nn.Module):
    """Presence classification plus masked severity regression.

    The severity term is masked to the positives on purpose. Severity is
    undefined for an issue that is not present — there is no meaningful "how
    blurry" for a sharp image — so the label is 0 by convention only. Training
    against those zeros would teach the head to regress toward 0 everywhere,
    which is a much easier objective than the real one and would swamp the
    signal from the minority of images where severity actually means something.
    """

    def __init__(self, pos_weight: torch.Tensor, severity_weight: float = 1.5):
        super().__init__()
        self.presence_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.severity_weight = severity_weight

    def forward(self, output: dict, target: dict) -> dict[str, torch.Tensor]:
        presence = self.presence_loss(output["presence_logits"], target["presence"])

        mask = target["presence"] > 0.5
        if mask.any():
            predicted = torch.sigmoid(output["severity_logits"])[mask]
            severity = nn.functional.mse_loss(predicted, target["severity"][mask])
        else:
            severity = torch.zeros((), device=presence.device)

        total = presence + self.severity_weight * severity
        return {
            "loss": total,
            # Detached copies for logging. Accumulating the graph-attached
            # tensors would keep every batch's autograd graph alive for the
            # whole epoch.
            "loss_value": total.detach(),
            "presence": presence.detach(),
            "severity": severity.detach(),
        }


def _batch_inputs(batch: dict, model: FocalNet, device: torch.device) -> dict:
    inputs = {}
    if model.config.use_image:
        # A batch carrying an embedding came from the cache, and the heads are
        # run directly from it. Dispatching on the batch rather than on a flag
        # keeps the two paths from drifting apart.
        if "embedding" in batch:
            inputs["embedding"] = batch["embedding"].to(device, non_blocking=True)
        else:
            inputs["image"] = batch["image"].to(device, non_blocking=True)
    if model.config.use_features:
        inputs["features"] = batch["features"].to(device, non_blocking=True)
    return inputs


def _apply(model: FocalNet, inputs: dict) -> dict:
    if "embedding" in inputs:
        return model.forward_from_embedding(**inputs)
    return model(**inputs)


def run_epoch(
    model: FocalNet,
    loader: DataLoader,
    criterion: FocalLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "presence": 0.0, "severity": 0.0}
    seen = 0
    all_probs, all_targets = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = _batch_inputs(batch, model, device)
            target = {
                "presence": batch["presence"].to(device),
                "severity": batch["severity"].to(device),
            }

            output = _apply(model, inputs)
            losses = criterion(output, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                # Multi-task losses can spike when a rare issue appears in a
                # batch; clipping keeps one such batch from wrecking the run.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            size = target["presence"].shape[0]
            seen += size
            totals["loss"] += float(losses["loss_value"]) * size
            totals["presence"] += float(losses["presence"]) * size
            totals["severity"] += float(losses["severity"]) * size

            if not training:
                all_probs.append(torch.sigmoid(output["presence_logits"]).cpu().numpy())
                all_targets.append(target["presence"].cpu().numpy())

    metrics = {key: value / max(seen, 1) for key, value in totals.items()}

    if all_probs:
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        metrics["mean_auc"] = float(np.mean([
            _auc(targets[:, i], probs[:, i]) for i in range(targets.shape[1])
        ]))
        metrics["mean_f1"] = float(np.mean([
            _f1(targets[:, i], probs[:, i] >= 0.5) for i in range(targets.shape[1])
        ]))

    return metrics


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-sum AUC; returns 0.5 when a split contains only one class."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = float((predicted & (labels > 0.5)).sum())
    if true_positive == 0:
        return 0.0
    precision = true_positive / float(predicted.sum())
    recall = true_positive / float((labels > 0.5).sum())
    return 2 * precision * recall / (precision + recall)


def train(config: TrainConfig, model_config: ModelConfig, smoke: bool = False) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = 200 if smoke else None
    print(f"device: {device}")

    train_set = load_split(
        config.generated_dir, "train",
        with_features=model_config.use_features,
        train=True, load_image=model_config.use_image, limit=limit,
    )
    val_set = load_split(
        config.generated_dir, "val",
        with_features=model_config.use_features,
        load_image=model_config.use_image, limit=limit,
    )
    print(f"train: {len(train_set)} images   val: {len(val_set)} images")

    workers = 0 if smoke else config.num_workers
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )

    # Phase A can run from cached embeddings, which is the difference between
    # roughly 40 minutes and under one minute per epoch on CPU.
    embedding_loaders = None
    if config.embeddings_dir and model_config.use_image:
        from training.dataset import load_embedding_split

        cached_train = load_embedding_split(config.embeddings_dir, "train", train_set, train=True)
        cached_val = load_embedding_split(config.embeddings_dir, "val", val_set)
        embedding_loaders = (
            DataLoader(cached_train, batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(cached_val, batch_size=config.batch_size, shuffle=False),
        )
        print(f"using cached embeddings from {config.embeddings_dir} for the frozen phase")

    model_config.n_features = len(FEATURE_NAMES)
    model = FocalNet(model_config).to(device)

    if model_config.use_features:
        mean, std = feature_statistics(train_set)
        model.set_feature_stats(mean.to(device), std.to(device))
        print("installed feature standardisation from the training split")

    pos_weight = positive_weights(train_set).to(device)
    print("pos_weight: " + ", ".join(f"{i}={w:.1f}" for i, w in zip(ISSUE_TYPES, pos_weight.tolist())))
    criterion = FocalLoss(pos_weight, config.severity_weight)

    history: list[dict] = []
    best_score, best_state, epochs_without_gain = -1.0, None, 0
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_phase(name: str, epochs: int, optimizer, scheduler=None, loaders=None) -> bool:
        nonlocal best_score, best_state, epochs_without_gain
        trainable, total = model.trainable_parameter_count()
        # ASCII only: Windows consoles default to a code page that mangles
        # anything else, and this is a CLI people run there.
        print(f"\n=== {name}: {epochs} epochs, {trainable:,}/{total:,} parameters trainable ===")
        phase_train, phase_val = loaders if loaders else (train_loader, val_loader)

        for epoch in range(1, epochs + 1):
            started = time.time()
            train_metrics = run_epoch(model, phase_train, criterion, device, optimizer)
            val_metrics = run_epoch(model, phase_val, criterion, device)
            if scheduler is not None:
                scheduler.step()

            record = {
                "phase": name, "epoch": epoch,
                "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
                "val_auc": val_metrics.get("mean_auc", 0.0), "val_f1": val_metrics.get("mean_f1", 0.0),
                "seconds": round(time.time() - started, 1),
            }
            history.append(record)
            print(
                f"  epoch {epoch:>2}  train {record['train_loss']:.4f}  "
                f"val {record['val_loss']:.4f}  AUC {record['val_auc']:.4f}  "
                f"F1 {record['val_f1']:.4f}  ({record['seconds']}s)"
            )

            # Selection is on validation AUC rather than loss. The loss mixes
            # two objectives on different scales, so a run can improve it by
            # getting better at severity regression while getting worse at the
            # detection the product actually depends on.
            if record["val_auc"] > best_score:
                best_score = record["val_auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= config.early_stopping_patience:
                    print(f"  early stopping — no gain in {epochs_without_gain} epochs")
                    return True
        return False

    # ---- Phase A: heads only ----
    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    stopped = run_phase(
        "Phase A (frozen backbone)",
        1 if smoke else config.head_epochs,
        torch.optim.AdamW(head_params, lr=config.head_lr, weight_decay=config.weight_decay),
        loaders=embedding_loaders,
    )

    # ---- Phase B: fine-tune the last blocks ----
    if not stopped and model_config.use_image:
        model.unfreeze_last_blocks(config.unfreeze_blocks)
        backbone_ids = {id(p) for p in model.features.parameters()}
        # Two parameter groups: the backbone moves two orders of magnitude more
        # slowly than the heads, because it starts from useful weights and the
        # heads do not.
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.features.parameters() if p.requires_grad],
                 "lr": config.backbone_lr},
                {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids],
                 "lr": config.finetune_head_lr},
            ],
            weight_decay=config.weight_decay,
        )
        epochs = 1 if smoke else config.finetune_epochs
        finetune_loaders = None
        if config.finetune_subset and config.finetune_subset < len(train_set):
            from torch.utils.data import Subset

            generator = torch.Generator().manual_seed(config.seed)
            chosen = torch.randperm(len(train_set), generator=generator)[: config.finetune_subset]
            finetune_loaders = (
                DataLoader(
                    Subset(train_set, chosen.tolist()), batch_size=config.batch_size,
                    shuffle=True, num_workers=workers, drop_last=True,
                ),
                val_loader,
            )
            print(f"fine-tuning on a {config.finetune_subset}-image subsample of the training split")

        run_phase(
            "Phase B (fine-tuning)", epochs, optimizer,
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs),
            loaders=finetune_loaders,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Everything needed to reproduce inference travels with the weights:
    # architecture config, the feature ordering the encoder expects, and the
    # standardisation statistics. A checkpoint that only carried tensors would
    # be silently wrong the moment FEATURE_NAMES changed order.
    checkpoint_path = out_dir / f"{config.model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config.to_dict(),
            "train_config": asdict(config),
            "issue_types": list(ISSUE_TYPES),
            "feature_names": list(FEATURE_NAMES),
            "best_val_auc": best_score,
            "history": history,
            "version": config.model_name,
        },
        checkpoint_path,
    )
    (out_dir / f"{config.model_name}_history.json").write_text(
        json.dumps({"history": history, "best_val_auc": best_score}, indent=2), encoding="utf-8"
    )

    print(f"\nBest validation AUC {best_score:.4f}")
    print(f"Wrote {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated-dir", default="dataset/generated")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--ablation", choices=["hybrid", "image", "features"], default="hybrid")
    parser.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    parser.add_argument("--epochs", type=int, default=None, help="override both phase lengths")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--embeddings", default=None,
        help="cached frozen-backbone embeddings, for a fast Phase A "
             "(build with: python -m training.cache_embeddings)",
    )
    parser.add_argument(
        "--finetune-subset", type=int, default=None,
        help="cap the training images used in Phase B",
    )
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    names = {"hybrid": "focal_cnn_v1", "image": "focal_cnn_image_only", "features": "focal_cnn_features_only"}
    config = TrainConfig(
        generated_dir=args.generated_dir,
        out_dir=args.out_dir,
        model_name=names[args.ablation],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embeddings_dir=args.embeddings,
        finetune_subset=args.finetune_subset,
    )
    if args.epochs:
        config.head_epochs = config.finetune_epochs = args.epochs
    if args.head_epochs is not None:
        config.head_epochs = args.head_epochs
    if args.finetune_epochs is not None:
        config.finetune_epochs = args.finetune_epochs

    model_config = ModelConfig(
        use_image=args.ablation in ("hybrid", "image"),
        use_features=args.ablation in ("hybrid", "features"),
        n_features=len(FEATURE_NAMES),
        pretrained=not args.no_pretrained,
    )

    train(config, model_config, smoke=args.smoke)


if __name__ == "__main__":
    main()
