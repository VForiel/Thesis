import json
try:
    with open('d:/THESIS/Thesis.ipynb', encoding='utf-8') as f:
        nb = json.load(f)
    if len(nb['cells']) > 75:
        cell = nb['cells'][75]
        with open('d:/THESIS/cell_content.txt', 'w', encoding='utf-8') as out:
            out.write(f"Cell Type: {cell['cell_type']}\n")
            out.write("Source:\n")
            out.write("".join(cell['source']))
    else:
        with open('d:/THESIS/cell_content.txt', 'w', encoding='utf-8') as out:
            out.write("Cell 75 not found")
except Exception as e:
    with open('d:/THESIS/cell_content.txt', 'w', encoding='utf-8') as out:
        out.write(f"Error: {e}")
