from flask import Flask, request, render_template
from pathlib import Path
import sys
import os
import pickle
import platform
import pytesseract
from PIL import Image
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))
# Configure Tesseract
# Windows: use the local Tesseract installation
# Render/Linux: use Tesseract from PATH
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
# Import PDF text extraction
from src.pdf_reader import extract_text_from_pdf
app = Flask(__name__)
# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "final_resume_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "final_tfidf_vectorizer.pkl"
# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
# Load TF-IDF vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        # Get uploaded file
        file = request.files.get("resume")
        if not file or file.filename == "":
            print("UPLOAD FILE:", file.filename if file else "NO FILE")
            error = "Please upload or capture a resume."
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )
        # Get filename
        filename = file.filename.lower()
        # Allowed file types
        allowed_extensions = (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png"
        )
        if not filename.endswith(allowed_extensions):
            error = (
                "Only PDF, JPG, JPEG, and PNG files are supported."
            )
            return render_template(
                "index.html",
                prediction=prediction,
                error=error
            )
        # Upload folder
        upload_folder = (
            PROJECT_ROOT /
            "webapp" /
            "uploads"
        )
        upload_folder.mkdir(
            parents=True,
            exist_ok=True
        )
        # Save uploaded file
        file_path = upload_folder / file.filename
        file.save(file_path)
        try:
            # -----------------------------
            # PDF Resume
            # -----------------------------
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(
                    file_path
                )
            # -----------------------------
            # Image Resume
            # -----------------------------
            else:
                image = Image.open(file_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                resume_text = pytesseract.image_to_string(
                    image
                )
            # Check extracted text
            if not resume_text.strip():
                error = (
                    "Could not extract text from this resume. "
                    "Please upload a clearer resume."
                )
            else:
                # Convert resume text into TF-IDF
                resume_tfidf = vectorizer.transform(
                    [resume_text]
                )
                # Predict category
                prediction = model.predict(
                    resume_tfidf
                )[0]
        except Exception as e:
            error = (
                f"Error processing resume: {e}"
            )
        finally:
            # Delete uploaded file after processing
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
# -----------------------------
# Start Flask application
# -----------------------------
if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )
    app.run(
        host="0.0.0.0",
        port=port
    )