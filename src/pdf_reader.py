import pymupdf
import pytesseract
from pathlib import Path


# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_pdf(file_path):
    file_path = Path(file_path)

    doc = pymupdf.open(str(file_path))
    text = ""

    # Try normal text extraction
    for page in doc:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text + "\n"

    # If no text found, use OCR
    if not text.strip():
        print("No text found. Using OCR...")

        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

            image = pix.tobytes("png")

            from PIL import Image
            from io import BytesIO

            pil_image = Image.open(BytesIO(image))

            ocr_text = pytesseract.image_to_string(pil_image)

            text += ocr_text + "\n"

    doc.close()

    return text
