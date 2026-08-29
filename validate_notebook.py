import json
with open(r'D:\Projects\DeepSeek-R1-Distill-Qwen-14B-Colab.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
print('Valid JSON, cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    print(f'  {i}: {c["cell_type"]}')