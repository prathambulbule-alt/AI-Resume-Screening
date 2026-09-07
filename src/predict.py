import pickle
import sys
from pathlib import Path
from io import BytesIO

import pypdf
import pymupdf
import pytesseract
from PIL import Image

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_pdf(file_path):
    file_path = Path(file_path)

    # First try normal PDF text extraction
    reader = pypdf.PdfReader(str(file_path))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # If no text found, use OCR
    if not text.strip():
        print("No text found. Using OCR...")

        doc = pymupdf.open(str(file_path))

        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

            image = pix.tobytes("png")
            pil_image = Image.open(BytesIO(image))

            ocr_text = pytesseract.image_to_string(pil_image)

            text += ocr_text + "\n"

        doc.close()

    return text


# Load trained model
with open("models/final_resume_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("models/final_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Check PDF argument
if len(sys.argv) < 2:
    print("Usage: python src\\predict.py <resume.pdf>")
    sys.exit(1)


pdf_path = Path(sys.argv[1])

if not pdf_path.exists():
    print(f"Error: File not found - {pdf_path}")
    sys.exit(1)


# Extract resume text
resume_text = extract_text_from_pdf(pdf_path)

if not resume_text.strip():
    print("Error: No text could be extracted from the PDF.")
    sys.exit(1)


print("\nResume text extracted successfully!")
print("Text length:", len(resume_text))


# Convert resume text into TF-IDF
resume_tfidf = vectorizer.transform([resume_text])


# Predict category
prediction = model.predict(resume_tfidf)

print("\nPredicted Job Category:", prediction[0])


# Decision score
decision_scores = model.decision_function(resume_tfidf)

predicted_index = list(model.classes_).index(prediction[0])
predicted_score = decision_scores[0][predicted_index]

print(f"Decision Score: {predicted_score:.4f}")