import cv2
import pytesseract
from PIL import Image

# If using Windows, set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """Preprocess the image for better OCR results."""
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance handwritten text
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return processed_img

def extract_text(image_path):
    """Extract handwritten text from an image using Tesseract OCR."""
    processed_img = preprocess_image(image_path)

    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(processed_img)

    # Perform OCR
    custom_config = r'--oem 3 --psm 6'  # Use OCR Engine Mode 3 and Page Segmentation Mode 6
    extracted_text = pytesseract.image_to_string(pil_img, config=custom_config)

    return extracted_text.strip()

if __name__ == "__main__":
    image_path = "hello.jpeg"  # Replace with your image path
    text = extract_text(image_path)
    print("Extracted Handwritten Text:\n", text)
