from flask import Flask, request, render_template
from pathlib import Path
import sys
import os
import pickle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.pdf_reader import extract_text_from_pdf
import pytesseract
from PIL import Image
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
        # Get the selected file from any of the 3 upload options
        file = request.files.get("resume")
        if not file or file.filename == "":
            print("UPLOAD FILE:", file.filename if file else "NO FILE")
            error = "Please upload or capture a resume."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )
        filename = file.filename.lower()
        allowed_extensions = (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png"
        )
        if not filename.endswith(allowed_extensions):
            error = "Only PDF, JPG, JPEG, and PNG files are supported."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )
        upload_folder = PROJECT_ROOT / "webapp" / "uploads"
        upload_folder.mkdir(parents=True, exist_ok=True)
        file_path = upload_folder / file.filename
        file.save(file_path)
        try:
            # PDF
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_path)
            # JPG / JPEG / PNG
            else:
                image = Image.open(file_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                resume_text = pytesseract.image_to_string(image)
            if not resume_text.strip():
                error = (
                    "Could not extract text from this resume. "
                    "Please upload a clearer resume."
                )
            else:
                resume_tfidf = vectorizer.transform(
                    [resume_text]
                )
                prediction = model.predict(
                    resume_tfidf
                )[0]
        except Exception as e:
            error = f"Error processing resume: {e}"
        finally:
            if file_path.exists():
                try:
                    os.remove(file_path)
                except Exception:
                    pass
    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )
if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )
    app.run(
        host="0.0.0.0",
        port=port
    )