import pandas as pd
root = r"C:\Users\Khang\Desktop\Final Project 2"

df = pd.read_csv(root + r"\data\processed\fx_features.csv", parse_dates=["date"])
print("=== fx_features.csv ===")
print("shape:", df.shape)
print("date range:", df["date"].min().date(), "->", df["date"].max().date())
print("\ncolumns (%d):" % len(df.columns))
for c in df.columns:
    print("  -", c, "|", str(df[c].dtype))

print("\n=== target_direction distribution ===")
if "target_direction" in df.columns:
    print(df["target_direction"].value_counts(dropna=False))
    print("mean (share UP):", round(df["target_direction"].mean(), 4))

print("\n=== NaN counts per column ===")
print(df.isna().sum()[df.isna().sum() > 0])

print("\n=== head(3) / tail(3) of date+targets ===")
cols = ["date"] + [c for c in df.columns if c.startswith("target")]
print(df[cols].head(3).to_string())
print(df[cols].tail(3).to_string())

print("\n\n=== feature_importance_preview.csv ===")
fi = pd.read_csv(root + r"\data\processed\feature_importance_preview.csv")
print("shape:", fi.shape)
print(fi.to_string())
