import torch
import cv2
import numpy as np
from model import CRNN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 26 + 10 + 1  # 26 letters + 10 digits + 1 blank (for CTC)
model = CRNN(input_height=32, hidden_size=256, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("handwriting_ocr.pth", map_location=device))
model.eval()

# Define character mapping (CTC assumes 0 is blank)
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
char_map = {i+1: c for i, c in enumerate(characters)}  # Start at index 1 (skip blank)

# Function to predict text
def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))  # Resize to match model input
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize
    img = img.to(device)

    with torch.no_grad():
        output = model(img)  # Shape: (T, N, C)
        output = output.log_softmax(2)  # Apply log_softmax
        decoded = output.argmax(2).cpu().numpy()  # Convert logits to class indices
        print("🔍 Decoded Output:", decoded)
        # 🔍 Debugging: Print raw model output
        print(f"Raw Output Shape: {output.shape}")
        print(f"Raw Output (First Time Step): {output[0, 0, :10]}")  # Print first 10 class probabilities

        preds = output.argmax(dim=2)  # Get highest probability character indices
        print(f"Predicted Indices: {preds}")  # Debugging: Print predicted indices
    
    # Decode predictions using CTC greedy decoding
    pred_text = []
    prev_char = None  # To handle CTC duplicate suppression
    print(f'{preds[0].cpu().numpy()}')

    for idx in preds[0].cpu().numpy():
        if idx in char_map:  # Ignore blanks & duplicate characters
            print(f'{idx}')
            pred_text.append(char_map[idx])
        prev_char = idx

    return "".join(pred_text)

# Example
print("Predicted Text:", predict("dataset/example/test2.jpg"))
