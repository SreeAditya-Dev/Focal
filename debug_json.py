import json
import sys

try:
    with open(r'D:\Projects\DeepSeek-R1-Distill-Qwen-14B-Colab.ipynb', encoding='utf-8') as f:
        data = f.read()
    print(f"File length: {len(data)} chars")
    nb = json.loads(data)
    print("JSON is valid!")
    print(f"Number of cells: {len(nb.get('cells', []))}")
except json.JSONDecodeError as e:
    print(f"JSON decode error: {e}")
    print(f"Error at line {e.lineno}, column {e.colno}")
    print(f"Error message: {e.msg}")
    # Show context around error
    lines = data.split('\n')
    start = max(0, e.lineno - 5)
    end = min(len(lines), e.lineno + 5)
    print("\nContext around error:")
    for i in range(start, end):
        marker = ">>> " if i == e.lineno - 1 else "    "
        print(f"{marker}{i+1:3}: {lines[i]}")
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback
    traceback.print_exc()