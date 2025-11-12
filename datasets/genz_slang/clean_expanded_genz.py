import pandas as pd

df = pd.read_csv("expanded_genz_slang.csv")

print(df.head())

filtered_df = df[~df['example'].str.startswith("Generate", na=False)]

print(f"Filtered out {len(df) - len(filtered_df)} rows that started with 'Generate'.")

df.to_csv("unclean_genz_slang_dataset.csv", index=False)

filtered_df.to_csv("clean_expanded_genz_slang_dataset.csv", index=False)