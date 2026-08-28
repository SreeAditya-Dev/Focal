"""Torch dataset over the generated corpus.

Two preprocessing choices here are specific to quality assessment and differ
from what an object-classification pipeline would do.

**The whole frame is resized to 224x224, not centre-cropped.** The ImageNet
convention — resize the short side to 256, then crop 224 — discards roughly a
quarter of the frame. For classification that is harmless, since the subject is
usually central. Here it is not: a scratch, an occluding patch or a blown corner
can sit anywhere, and cropping it away turns a correctly-labelled defective
image into a mislabelled clean one. Squashing the aspect ratio costs less than
losing the edges.

**Augmentation is limited to flips.** Every transform an ordinary pipeline
applies — brightness jitter, contrast jitter, blur, JPEG noise, random resized
crop — is itself one of the degradations being detected, and would silently
rewrite the label it was trained against. Colour jitter on an image labelled
"correctly exposed" produces an underexposed image still labelled correct.
Scale changes are equally unsafe: magnifying a crop enlarges the blur kernel
and the noise grain along with it. Flips are the only geometric operation that
leaves all six labels intact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from focal_ml.constants import CNN_INPUT_SIZE, ISSUE_TYPES
from focal_ml.features import FEATURE_NAMES
from focal_ml.utils import imread_bgr

#: ImageNet statistics — required, because the backbone's pretrained filters
#: were fitted in this normalised space.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FocalDataset(Dataset):
    """Yields (image, features, presence, severity) for one manifest split."""

    def __init__(
        self,
        manifest,
        features_frame=None,
        root: str | Path = "dataset/generated",
        train: bool = False,
        load_image: bool = True,
        size: int = CNN_INPUT_SIZE,
    ):
        self.root = Path(root)
        self.train = train
        self.load_image = load_image
        self.size = size

        self.paths = manifest["image_path"].tolist()
        self.presence = manifest[[f"{issue}_present" for issue in ISSUE_TYPES]].to_numpy(np.float32)
        self.severity = manifest[[f"{issue}_severity_score" for issue in ISSUE_TYPES]].to_numpy(np.float32)

        if features_frame is not None:
            # Reindexed by path rather than assumed to be row-aligned. The two
            # tables are produced by separate scripts and separate parallel runs;
            # a silent misalignment here would pair every image with another
            # image's measurements and be nearly impossible to spot downstream.
            indexed = features_frame.set_index("image_path")
            missing = set(self.paths) - set(indexed.index)
            if missing:
                raise ValueError(f"{len(missing)} images have no extracted features, e.g. {sorted(missing)[:3]}")
            self.features = indexed.loc[self.paths, list(FEATURE_NAMES)].to_numpy(np.float32)
        else:
            self.features = None

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, index: int) -> torch.Tensor:
        import cv2

        image = imread_bgr(self.root / self.paths[index])
        if image is None:
            # A corpus image that will not decode is a generation bug, not a
            # condition to model around — but failing the whole epoch for one
            # bad file is worse, so substitute grey and keep going.
            image = np.full((self.size, self.size, 3), 128, dtype=np.uint8)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.train and np.random.random() < 0.5:
            image = image[:, ::-1]

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "presence": torch.from_numpy(self.presence[index]),
            "severity": torch.from_numpy(self.severity[index]),
        }
        if self.load_image:
            sample["image"] = self._load_image(index)
        if self.features is not None:
            sample["features"] = torch.from_numpy(self.features[index])
        return sample


def load_split(
    generated_dir: str | Path,
    split: str,
    *,
    with_features: bool = True,
    train: bool = False,
    load_image: bool = True,
    limit: int | None = None,
) -> FocalDataset:
    """Build a dataset for one split from the manifest and feature table."""
    import pandas as pd

    generated_dir = Path(generated_dir)
    manifest = pd.read_csv(generated_dir / "manifest.csv")
    manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    if limit:
        manifest = manifest.head(limit)

    features_frame = None
    if with_features:
        parquet = generated_dir / "features.parquet"
        csv = generated_dir / "features.csv"
        path = parquet if parquet.exists() else csv
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Run: python -m dataset.extract_features"
            )
        features_frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    return FocalDataset(
        manifest, features_frame, root=generated_dir, train=train, load_image=load_image
    )


def positive_weights(dataset: FocalDataset, cap: float = 12.0) -> torch.Tensor:
    """Per-issue ``pos_weight`` for ``BCEWithLogitsLoss``.

    Each issue appears in roughly a sixth of the corpus, so an unweighted loss
    is minimised well by predicting "absent" for everything. Weighting the
    positive term by the negative/positive ratio restores the balance.

    A ``WeightedRandomSampler`` is deliberately not used instead: with six
    co-occurring labels per image there is no single quantity to balance the
    sampling on, and oversampling to fix one issue's ratio distorts the other
    five. Reweighting the loss addresses each issue independently.

    The cap keeps a rare issue from producing a gradient so large it destabilises
    the shared trunk.
    """
    positives = dataset.presence.sum(axis=0)
    negatives = len(dataset) - positives
    weights = np.divide(
        negatives, positives, out=np.ones_like(positives), where=positives > 0
    )
    return torch.from_numpy(np.clip(weights, 0.5, cap).astype(np.float32))


def feature_statistics(dataset: FocalDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of the log-compressed features.

    Computed on the training split only — deriving them from the full corpus
    would leak validation and test distribution into training.
    """
    if dataset.features is None:
        raise ValueError("dataset was built without features")
    values = torch.from_numpy(dataset.features)
    from focal_ml.model.architecture import FocalNet

    compressed = FocalNet.transform_features(values)
    return compressed.mean(dim=0), compressed.std(dim=0)
