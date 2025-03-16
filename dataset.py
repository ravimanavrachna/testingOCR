import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import cv2
import os

# Define vocabulary (Lowercase letters, digits, and a blank token
vocab = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
char_to_idx = {char: i for i, char in enumerate(vocab)}

class HandwritingDataset(Dataset):
    def __init__(self, image_folder, label_file):  # ✅ Fixed __init__
        self.image_folder = image_folder
        self.labels = self.load_labels(label_file)
        self.image_files = list(self.labels.keys())

    def load_labels(self, label_file):
        """Load image labels from a text file"""
        labels = {}
        with open(label_file, "r") as file:
            for line in file:
                print(f'{line}')
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    labels[parts[0]] = parts[1]
                else:
                     labels[parts[0]] = ''  # { "image1.jpg": "hello" }
        return labels

    def __len__(self):  # ✅ Fixed __len__
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize

        label_text = self.labels[self.image_files[idx]]
        label = encode_label(label_text)  # ✅ Use fixed vocabulary mapping

        #print(f'Label: {label_text} -> Encoded: {label}')  # Debugging
        return image, label

def collate_batch(batch):
    images, labels = zip(*batch)

    # Stack images into a single batch tensor
    images = torch.stack(images, dim=0)

    # Convert labels to tensors and remove padding
    cleaned_labels = [torch.tensor(label, dtype=torch.long) if isinstance(label, list) else label for label in labels]
    target_lengths = torch.tensor([len(label) for label in cleaned_labels], dtype=torch.long)

    # **Concatenate into a single 1D tensor (CTC requires this)**
    labels = torch.cat(cleaned_labels, dim=0)

    return images, labels, target_lengths

def encode_label(text):
    blank_idx = 0  # Set blank token index
    return torch.tensor([char_to_idx.get(c, blank_idx) for c in text], dtype=torch.long)


# Example Usage
train_dataset = HandwritingDataset("dataset/processed", "dataset/labels.txt")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

print(f"✅ Dataset loaded successfully! Total samples: {len(train_dataset)}")
