import json
import re

nb_path = 'Thesis.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'analysis.' in source or '.plot(' in source or '.plot_' in source:
             print(f"--- Cell {i} ---")
             # Find lines with interesting calls
             for line in source.split('\n'):
                 if 'analysis.' in line or '.plot' in line:
                     print(line.strip())
