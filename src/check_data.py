import pandas as pd

file_path = "data/Resume.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nCategories:")
print(df["Category"].value_counts())