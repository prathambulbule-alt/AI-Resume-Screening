import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# Load dataset
data = pd.read_csv("data/cleaned_resume.csv")

X = data["Resume_str"]
y = data["Category"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load final model
with open("models/final_resume_model.pkl", "rb") as f:
    model = pickle.load(f)


# Load final vectorizer
with open("models/final_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Convert test data to TF-IDF
X_test_tfidf = vectorizer.transform(X_test)


# Predict
y_pred = model.predict(X_test_tfidf)


# Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


# Create plot
plt.figure(figsize=(14, 14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    xticks_rotation=90
)

plt.title("Resume Classification - Confusion Matrix")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png", dpi=300)

print("Confusion Matrix saved successfully!")

plt.show()