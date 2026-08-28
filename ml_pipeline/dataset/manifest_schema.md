# Manifest schema

`generate_synthetic.py` writes `manifest.csv` and `manifest.parquet` to the
output directory. One row per generated image; 30 columns.

## Identity and provenance

| Column | Type | Notes |
|---|---|---|
| `image_path` | str | Relative to the manifest, e.g. `train/coco_00042_000000581615__blur.jpg` |
| `base_image_id` | str | `{source}_{index:05d}_{stem}` — every variant of a scene shares this |
| `source` | str | `div2k`, `div2k_train`, `coco`, or `local` |
| `split` | str | `train` / `val` / `test`, assigned per **base image** (see below) |
| `variant` | str | `clean0`, `clean1`, `blur`, …, `combo0_blur_noise` |
| `width`, `height` | int | Dimensions after canonical resize |
| `degradation_params` | str (JSON) | List of `{issue, method, severity_score, params}` — the exact transform applied |

## Labels

For each of the six issue types in `focal_ml.constants.ISSUE_TYPES`
(`blur`, `underexposure`, `overexposure`, `noise`, `corruption`, `defect`):

| Column | Type | Notes |
|---|---|---|
| `{issue}_present` | 0/1 | Multi-label target for the CNN's presence heads |
| `{issue}_severity_score` | float 0-1 | **The ground truth.** Target for the severity regression heads |
| `{issue}_severity_bucket` | 0-3 | `none`/`low`/`medium`/`high`, derived from the score — never stored independently, so the two cannot disagree |

Plus the aggregate:

| Column | Type | Notes |
|---|---|---|
| `is_clean` | 0/1 | No degradation applied |
| `n_issues` | int | 0, 1, or 2 |
| `quality_score` | float 0-100 | `focal_ml.constants.compute_quality_score` over the severities |
| `quality_label` | str | `EXCELLENT` / `ACCEPTABLE` / `POOR` / `UNUSABLE` |

`quality_score` is computed by the **same function the API calls at inference**,
so the predicted and ground-truth scores are directly comparable — that is what
makes the regression metrics in Phase 5 meaningful rather than arbitrary.

## Why severity is continuous

Buckets are a presentation detail for the API response. The score is the
label because a 0-1 target lets the model express "borderline" rather than
being forced to pick a side of a threshold it cannot see. Bucket boundaries
live in `SEVERITY_SCORE_RANGES` and are applied identically everywhere via
`severity_bucket_from_score`.

For **exposure specifically**, the severity score is measured from the
*achieved* mean luma of the output image rather than from the gamma that was
requested. The same gamma darkens a bright scene and a dim one by visibly
different amounts, so a parameter-derived label would not describe what the
image actually looks like.

## Split integrity

Splits are assigned **per base image, before any degradation is applied**, and
stratified by source. Every variant of a given photograph lands in the same
split.

Splitting after generation would put a blurred copy and a noisy copy of the
same scene on opposite sides of the train/test boundary. A model can then score
well by recognising the scene rather than the defect, and the reported
generalisation would be fiction.

## What is *not* in here

`dataset/invalid_samples/` holds files that are genuinely undecodable —
zero-byte, truncated, random bytes, text with an image extension. These are
**not** training data and have no manifest rows. They exist to exercise the
API's rejection path, and are deliberately distinct from the `corruption`
class, which decodes perfectly well and merely looks damaged.
