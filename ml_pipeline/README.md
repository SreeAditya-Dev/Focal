# Focal ML pipeline

Dataset generation, feature engineering, model training and evaluation.
All commands are run **from this directory** (`ml_pipeline/`).

## Install

```bash
pip install -e ".[data]"      # dataset generation
pip install -e ".[train]"     # + training and evaluation (pulls torch)
```

The `focal_ml` package is imported directly by the backend, so the backend
image installs this same project with the `[torch]` extra.

## Phase 1 — build the corpus

### 1. Acquire and screen clean base images

```bash
python -m dataset.download_sources                        # DIV2K valid + COCO val2017, ~1.2 GB
python -m dataset.download_sources --sources div2k coco div2k_train   # + 3.3 GB for 800 more
python -m dataset.download_sources --sources --local "C:/my/photos"   # your own images, no download
```

Downloads resume if interrupted, and archives are deleted after extraction
unless `--keep-archives` is passed.

Every candidate is screened before being accepted as "clean", because a soft or
badly exposed source image silently poisons both the no-issue class and every
degradation derived from it. Measured on a 512px copy so thresholds mean the
same thing across sources:

| Check | Threshold |
|---|---|
| Resolution | ≥ 640 × 480 |
| Sharpness (variance of Laplacian) | ≥ 150 |
| Mean luma | 70 – 185 |
| RMS contrast | ≥ 25 |
| Estimated noise σ | ≤ 4.0 |

Writes `dataset/raw/base_index.json`.

### 2. Generate the labelled corpus

```bash
python -m dataset.generate_synthetic                 # full run
python -m dataset.generate_synthetic --limit 20      # smoke test
```

Ten variants per base image — 2 clean, 6 single-issue (one per type), 2
multi-issue — so ~2,400 base images yields ~24,000 labelled images with clean
held at 20%.

Writes `dataset/generated/{split}/*.jpg`, `manifest.csv`, `manifest.parquet`,
`generation_meta.json`, and the `dataset/invalid_samples/` fixtures used by the
backend's rejection tests. See [`dataset/manifest_schema.md`](dataset/manifest_schema.md).

Generation is **bit-for-bit reproducible**: each image is seeded from
`(--seed, image index)`, so output does not depend on `--workers`.

### Inspect the degradations

```bash
python -m dataset.degradations --demo path/to/photo.jpg --out dataset/demo
```

Renders every issue × method × severity (48 images) plus the clean original,
and prints the severity each one actually achieved.

## Phase 2 — classical features and rules

### Extract features over the corpus

```bash
python -m dataset.extract_features --report
```

Writes `dataset/generated/features.parquet` (47 features per image) and, with
`--report`, ranks each feature by per-issue AUC so you can see which measurements
actually carry signal.

### Refit the rule thresholds

```bash
python -m training.fit_rules
```

The shipped thresholds are argued from the physical meaning of each measurement
(mean luma 95 is where shadow detail starts to go; a blockiness ratio of 1.0
means no preferential 8×8 structure). This refits them to percentiles of the
training split and prints the before/after validation F1 so the change is
visible rather than assumed. Which features drive which issue, and how they
combine, are *not* refit — that structure encodes reasoning that should not be
silently rewritten by whatever correlates best in one corpus.

### Run the tests

```bash
python -m pytest tests/ -q          # ~2 min; every assertion is directional
```

### The 47 features

| Group | Measurements | Targets |
|---|---|---|
| Sharpness | Laplacian variance, contrast-normalised ratio, Tenengrad, Canny edge density, FFT high-frequency ratio, per-tile distribution | blur |
| Exposure | V-channel mean/std/percentiles, shadow & highlight clipping, RMS contrast, dynamic range, histogram entropy | under/overexposure |
| Noise | Immerkaer σ, flat-region σ, chroma σ, median-residual MAD, impulse ratio, texture ratio | noise |
| Texture | GLCM contrast/homogeneity/energy/correlation, LBP uniformity | over-smoothing, detail loss |
| Colour | saturation mean/std, Hasler–Süsstrunk colourfulness, grey fraction, channel edge correlation | desaturation, chroma faults |
| Artifacts | blockiness, flat-block fraction, block jump, row/column discontinuity, largest uniform region, byte entropy | corruption |
| Defects | radial falloff, linear structure, local contrast spread | vignetting, scratches |

Extraction takes ~70 ms for a canonical 768 px image and ~180 ms including the
resize from a 4000 px upload, which leaves room for CNN inference inside a
synchronous request.

### Two findings worth recording

**Noise estimators are confounded by texture**, so the primary estimator reads
the noise floor from the flattest blocks in the image, where by construction
almost all remaining variation is noise. A photograph of gravel otherwise reads
as extremely noisy.

**Uneven sharpness does not identify a localised defect.** The obvious way to
catch a lens smudge is to notice that some tiles are much softer than others —
but measurement shows uniform blur drives that statistic down just as far
(0.14, against 0.12 for a smudge and 0.29 for a clean frame), because blur
flattens weakly-textured tiles faster than strongly-textured ones. Using it
alone made motion blur rank as a defect. What actually separates the two is
that *the sharpest tiles survive a smudge* — they retain 97% of their clean
sharpness, against 1% under Gaussian blur. The rule layer therefore expresses
this as a conjunction (`RampGroup`), and only severe smudges clear it; mild
ones overlap the clean population outright and are left to the CNN.

## Design notes

**Why synthetic.** Owning every transform parameter means the labels are exact
rather than annotated, and severity is a continuous quantity rather than a
crowd-sourced opinion. The cost is a domain gap against real degradation, which
Phase 5 measures explicitly against a set of real photographs.

**Resize before degrading, not after.** Every image is normalised to a 768px
long side *first*. Downscaling a noisy image averages the noise away, so
degrading at native resolution and then resizing would leave labels describing
an image that no longer exists. The same constant governs feature extraction at
inference, so absolute-scale measurements like Laplacian variance transfer from
training images to user uploads.

**Physically ordered stacking.** Multi-issue variants apply degradations in
capture order — exposure, optics, sensor defects, sensor noise, compression —
rather than at random. Adding noise after blur produces a combination no camera
can generate.

**Non-uniform severity prior.** Buckets are sampled 45/35/20 (low/medium/high).
Among photographs that have a problem at all, mild problems vastly outnumber
catastrophic ones; sampling uniformly pushes half the corpus into the POOR band.

**Clean variants include benign transforms.** One of the two clean variants per
image is horizontally flipped and re-encoded at JPEG 88–96, still labelled
clean. Without it the model learns that ordinary compression is a defect and
flags every re-saved photograph.

## Roadmap

- [x] **Phase 1** — dataset acquisition and synthetic degradation
- [x] **Phase 2** — classical features (`focal_ml/features/`) and rule layer (`focal_ml/fusion/rules.py`)
- [ ] **Phase 3** — CNN training (`training/`)
- [ ] **Phase 4** — fusion, Grad-CAM, calibration (`focal_ml/fusion/`, `focal_ml/inference/`)
- [ ] **Phase 5** — evaluation (`evaluation/`)
