import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_image(image_path):
    """Convert image to grayscale, resize, and apply thresholding"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))  # Resize to fixed size
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Binarization
    return img

def process_dataset(input_folder, output_folder):
    """Apply preprocessing to all images in dataset"""
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            processed_img = preprocess_image(img_path)
            cv2.imwrite(os.path.join(output_folder, filename), processed_img)

# Example usage
process_dataset("dataset/raw", "dataset/processed")
