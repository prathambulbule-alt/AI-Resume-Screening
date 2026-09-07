import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Linear SVM",
    "Naive Bayes"
]

accuracy = [
    65.79,
    71.63,
    54.73
]

plt.figure(figsize=(9, 6))

bars = plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)

print("Model comparison graph saved successfully!")

plt.show()