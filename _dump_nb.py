import json, sys
path = r"C:\Users\Khang\Desktop\Final Project 2\notebooks\04_feature_engineering.ipynb"
nb = json.load(open(path, encoding="utf-8"))
print("TOTAL CELLS:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    if i < 12:
        continue
    src = "".join(c["source"])
    print("===== CELL", i, "[" + c["cell_type"] + "] =====")
    print(src)
    print()
