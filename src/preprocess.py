import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Keep only required columns
df = df[["Resume_str", "Category"]]

# Remove missing values
df = df.dropna(subset=["Resume_str", "Category"])
df["Resume_str"] = df["Resume_str"].fillna("").astype(str)
df["Category"] = df["Category"].fillna("").astype(str)

# Clean resume text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)       # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Remove numbers/symbols
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

df["Resume_str"] = df["Resume_str"].apply(clean_text)

df = df[df["Resume_str"].str.strip() != ""]

# Remove duplicate resumes
df = df.drop_duplicates()

print("Cleaned Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCategories:")
print(df["Category"].value_counts())

# Save cleaned dataset
df.to_csv("data/cleaned_resume.csv", index=False)

print("\nCleaned dataset saved successfully!")