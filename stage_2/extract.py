import cv2
import pytesseract

def extract_details(image_path):
    """
    Extracts text from an image using Tesseract OCR.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        str: Extracted text from the image.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better OCR accuracy
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Extract text using Tesseract
    text = pytesseract.image_to_string(thresh)

    return text
