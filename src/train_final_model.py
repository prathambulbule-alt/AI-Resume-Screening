import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Load cleaned dataset
df = pd.read_csv("data/cleaned_resume.csv")

X = df["Resume_str"]
y = df["Category"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)


# Train Linear SVM
model = LinearSVC(
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# Evaluate model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nFinal Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save final model
with open("models/final_resume_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("models/final_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("\nFinal model saved successfully!")
print("Final vectorizer saved successfully!")