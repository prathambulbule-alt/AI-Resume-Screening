from flask import Flask, request, render_template
from pathlib import Path
import sys
import os

# Project root ko Python path me add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_reader import extract_text_from_pdf

import pickle

app = Flask(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "final_resume_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "final_tfidf_vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":

        if "resume" not in request.files:
            error = "Please upload a resume PDF."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )

        file = request.files["resume"]

        if file.filename == "":
            error = "Please select a PDF file."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )

        if not file.filename.lower().endswith(".pdf"):
            error = "Only PDF files are supported."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )

        upload_folder = PROJECT_ROOT / "webapp" / "uploads"
        upload_folder.mkdir(exist_ok=True)

        file_path = upload_folder / file.filename
        file.save(file_path)

        try:
            resume_text = extract_text_from_pdf(file_path)

            if not resume_text.strip():
                error = "Could not extract text from this resume."
            else:
                resume_tfidf = vectorizer.transform([resume_text])
                prediction = model.predict(resume_tfidf)[0]

        except Exception as e:
            error = f"Error processing resume: {e}"

        finally:
            if file_path.exists():
                os.remove(file_path)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)