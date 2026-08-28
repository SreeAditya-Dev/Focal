"""Quick live test: run FocalPredictor on a handful of held-out generated test images."""
from __future__ import annotations

import json
from pathlib import Path

from focal_ml.inference.predictor import FocalPredictor

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
TEST_DIR = ROOT / "dataset" / "generated" / "test"

predictor = FocalPredictor(
    model_path=str(MODELS / "focal_cnn_v1.pt"),
    calibration_path=str(MODELS / "calibration_v1.json"),
    rules_path=str(MODELS / "rules_v1.json"),
    device="cpu",
)
print("Health:", json.dumps(predictor.health(), indent=2))
print("Warmup...")
predictor.warmup()
print("Warmup done\n" + "=" * 80)

samples = [
    "coco_00087_000000531495__clean0.jpg",
    "coco_00087_000000531495__blur.jpg",
    "coco_00087_000000531495__overexposure.jpg",
    "coco_00087_000000531495__underexposure.jpg",
    "coco_00087_000000531495__noise.jpg",
    "coco_00087_000000531495__defect.jpg",
    "coco_00087_000000531495__corruption.jpg",
    "coco_00087_000000531495__combo0_underexposure_noise.jpg",
    "coco_00093_000000184384__clean0.jpg",
    "coco_00093_000000184384__blur.jpg",
    "coco_00093_000000184384__defect.jpg",
]

for name in samples:
    path = TEST_DIR / name
    if not path.exists():
        print(f"MISSING: {name}")
        continue
    raw = path.read_bytes()
    result = predictor.analyse(
        raw,
        raw_bytes=raw,
        include_heatmap=False,
        uncertainty=True,
    )
    d = result.to_dict()
    print(f"\n--- {name} ---")
    print(f"  score={d['quality_score']:.2f}  label={d['quality_label']}  "
          f"size={d['width']}x{d['height']}  model={d['model_version']}")
    print(f"  time_ms={d['processing_time_ms']:.1f}  "
          f"timings={ {k: round(v, 1) for k, v in d['timings_ms'].items()} }")
    print(f"  summary: {d['summary']}")
    if d['issues']:
        for issue in d['issues']:
            unc = next((u for u in (d.get('uncertainty') or []) if u['issue'] == issue['type']), None)
            unc_s = f" ±{unc['std'] * 100:.1f}%" if unc else ""
            print(f"    - {issue['type']:>10}  sev={issue['severity_score']:.2f}  "
                  f"conf={issue['confidence']:.2f}{unc_s}  [{issue['severity']}]")
    else:
        print("    (no issues detected)")
    if d.get('uncertainty'):
        print("  uncertainty:")
        for u in d['uncertainty']:
            print(f"    - {u['issue']:>10}  mean={u['mean']:.2f}  std={u['std']:.3f}  flagged={u['flagged']}")
