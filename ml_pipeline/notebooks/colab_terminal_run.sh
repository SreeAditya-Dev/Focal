# Paste this into the Colab terminal after bundle.zip has arrived at /content/bundle.zip
set -e
cd /content

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Get the package (code only — a few hundred KB, separate from the data bundle)
if [ ! -d ml_pipeline ]; then
  echo "ml_pipeline/ not found. Zip it locally and PUT it to this same tunnel as package.zip, then:"
  echo "  unzip -q package.zip"
fi
cd ml_pipeline
pip install -q -e ".[train]"

mkdir -p dataset/generated
unzip -q /content/bundle.zip -d dataset/generated
python3 -c "
import pandas as pd
m = pd.read_csv('dataset/generated/manifest.csv')
f = pd.read_parquet('dataset/generated/features.parquet')
print(f'{len(m)} images, {len(f)} feature rows')
assert len(f) >= len(m) * 0.95, 'feature table incomplete'
print('OK')
"

python3 -m training.fit_rules --out models/rules_v1.json

python3 -m training.train \
  --head-epochs 10 \
  --finetune-epochs 15 \
  --batch-size 64 \
  --num-workers 2

python3 -m training.calibrate \
  --model models/focal_cnn_v1.pt \
  --rules models/rules_v1.json \
  --tune-fusion

python3 -m evaluation.evaluate --ablation

echo "### DONE — models/ and evaluation/reports/ are ready to zip and pull back ###"
