"""The learned half of the hybrid.

A MobileNetV3-Small backbone with two multi-label heads — one asking whether
each issue is present, one asking how bad it is.

**Why the classical features are also fed into the network.** The obvious design
is a pure CNN, with the hand-built features used only to cross-check its output
at decision time. That throws away the main advantage of having them. The CNN
sees a 224x224 image; the features are measured on the full 768px frame. Noise
sigma, JPEG blockiness on the 8x8 grid, and impulse ratio all live in exactly
the detail that downscaling to 224 destroys — the network is structurally unable
to recompute them, no matter how well it is trained. Feeding them in as a second
input branch gives the model evidence it otherwise cannot obtain.

The two branches are complementary rather than redundant: the features carry
fine-scale measurement the image branch has lost, and the image carries spatial
layout the 47 scalars cannot express (*where* the frame is damaged, and whether
softness is localised or global).

``use_features`` and ``use_image`` can each be disabled, which is what makes the
three-way ablation in Phase 5 — image-only, features-only, hybrid — a comparison
between the same architecture and training loop rather than between three
different programs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from focal_ml.constants import CNN_INPUT_SIZE, ISSUE_TYPES

#: Channel count MobileNetV3-Small's final convolution emits.
BACKBONE_DIM = 576


@dataclass
class ModelConfig:
    use_image: bool = True
    use_features: bool = True
    n_features: int = 47
    feature_embedding_dim: int = 64
    trunk_dim: int = 256
    dropout: float = 0.2
    pretrained: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class FocalNet(nn.Module):
    """Multi-label presence + severity prediction from an image and its features."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.n_issues = len(ISSUE_TYPES)

        if not (self.config.use_image or self.config.use_features):
            raise ValueError("at least one of use_image / use_features must be enabled")

        fused_dim = 0

        if self.config.use_image:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.config.pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            # Keep only the convolutional trunk; the ImageNet classifier is
            # discarded. `features[-1]` is the Grad-CAM target in Phase 4.
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            fused_dim += BACKBONE_DIM

        if self.config.use_features:
            # Standardisation statistics live in the module as buffers so they
            # are saved and loaded with the weights. Keeping them anywhere else
            # invites the classic serving bug where a model is restored with
            # different normalisation than it was trained with.
            self.register_buffer("feature_mean", torch.zeros(self.config.n_features))
            self.register_buffer("feature_std", torch.ones(self.config.n_features))
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.config.n_features, self.config.feature_embedding_dim),
                nn.BatchNorm1d(self.config.feature_embedding_dim),
                nn.Hardswish(),
                nn.Linear(self.config.feature_embedding_dim, self.config.feature_embedding_dim),
                nn.Hardswish(),
            )
            fused_dim += self.config.feature_embedding_dim

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, self.config.trunk_dim),
            nn.Hardswish(),
            # This dropout does double duty: regularisation during training, and
            # the source of the Monte-Carlo uncertainty estimate at inference,
            # where it is left active across repeated forward passes.
            nn.Dropout(self.config.dropout),
        )

        self.presence_head = nn.Linear(self.config.trunk_dim, self.n_issues)
        # Severity is bounded 0-1, so the head emits a logit and the caller
        # applies a sigmoid. Predicting it unbounded and clamping would put the
        # loss on a different scale than the label.
        self.severity_head = nn.Linear(self.config.trunk_dim, self.n_issues)

    # ------------------------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def transform_features(features: torch.Tensor) -> torch.Tensor:
        """Compress the dynamic range before standardisation.

        The feature vector mixes quantities spanning six orders of magnitude —
        Laplacian variance reaches the thousands while clipping fractions live
        in [0, 1]. A signed log is monotonic, so it preserves every ordering the
        ramps and the model rely on, while pulling the heavy tails in far enough
        that standardisation is meaningful rather than being dominated by a
        handful of extreme images.
        """
        return torch.sign(features) * torch.log1p(features.abs())

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install training-set standardisation statistics (already log-space)."""
        if not self.config.use_features:
            return
        self.feature_mean.copy_(mean)
        # A constant feature would otherwise divide by zero and emit NaN.
        self.feature_std.copy_(torch.clamp(std, min=1e-6))

    def _encode_features(self, features: torch.Tensor) -> torch.Tensor:
        normalised = (self.transform_features(features) - self.feature_mean) / self.feature_std
        return self.feature_encoder(normalised)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, image: torch.Tensor | None = None, features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        parts = []

        if self.config.use_image:
            if image is None:
                raise ValueError("model was configured with use_image but no image was given")
            parts.append(self.pool(self.features(image)).flatten(1))

        if self.config.use_features:
            if features is None:
                raise ValueError("model was configured with use_features but none were given")
            parts.append(self._encode_features(features))

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        hidden = self.trunk(fused)

        return {
            "presence_logits": self.presence_head(hidden),
            "severity_logits": self.severity_head(hidden),
        }

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pooled backbone embedding, before fusion with the feature branch."""
        if not self.config.use_image:
            raise ValueError("this model has no image branch")
        return self.pool(self.features(image)).flatten(1)

    def forward_from_embedding(
        self, embedding: torch.Tensor | None = None, features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Run the heads from a precomputed backbone embedding.

        While the backbone is frozen its output is a fixed function of the
        image, so recomputing it every epoch does identical work repeatedly —
        and that convolution is essentially the entire cost of a training step.
        Caching the embeddings once turns the frozen-backbone phase into
        training a small MLP.

        Only valid while the backbone really is frozen. Once fine-tuning
        reopens it the embeddings change every step and ``forward`` must be
        used instead.
        """
        parts = []
        if self.config.use_image:
            if embedding is None:
                raise ValueError("model uses the image branch but no embedding was given")
            parts.append(embedding)
        if self.config.use_features:
            if features is None:
                raise ValueError("model uses the feature branch but none were given")
            parts.append(self._encode_features(features))

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        hidden = self.trunk(fused)
        return {
            "presence_logits": self.presence_head(hidden),
            "severity_logits": self.severity_head(hidden),
        }

    @torch.no_grad()
    def predict(
        self, image: torch.Tensor | None = None, features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Probabilities and bounded severities, for inference."""
        output = self.forward(image, features)
        return {
            "presence": torch.sigmoid(output["presence_logits"]),
            "severity": torch.sigmoid(output["severity_logits"]),
        }

    # ------------------------------------------------------------------
    # Transfer-learning schedule
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Phase A: train the heads against fixed ImageNet representations.

        Starting with the backbone unfrozen would let the large, randomly
        initialised head gradients flow back and destroy the pretrained filters
        in the first few steps, which is the usual way transfer learning is
        wasted.
        """
        if not self.config.use_image:
            return
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count: int = 2) -> None:
        """Phase B: reopen the final blocks for fine-tuning.

        Only the last blocks. Early layers encode edges and colour opponency
        that are as valid for quality assessment as for classification, while
        the late layers encode object semantics that need to be repurposed.
        """
        if not self.config.use_image:
            return
        for block in list(self.features)[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, sum(p.numel() for p in self.parameters())


def build_model(config: ModelConfig | None = None) -> FocalNet:
    return FocalNet(config)


def expected_input_size() -> int:
    return CNN_INPUT_SIZE
