import json

notebook_path = 'd:/PHISE/numerical_simulation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the ML code
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if it contains the commented out code
        if any('TODO: refactor the following code' in line for line in source):
            print("Found ML cell")
            # Replace the source
            new_source = [
                "from phise.modules import ml\n",
                "\n",
                "DATASET = ml.get_dataset(10_000)\n",
                "print(DATASET.shape)\n",
                "MODEL = ml.get_model(input_shape=DATASET.shape[1]-14)\n",
                "MODEL.summary()\n",
                "ml.train_model(MODEL, DATASET, plot=True)\n",
                "ml.test_model(MODEL, DATASET)"
            ]
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {notebook_path}")
else:
    print("ML cell not found")
