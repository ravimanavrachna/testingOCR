import torch
import torch.nn as nn
import torch.optim as optim
from model import CRNN
from dataset import train_loader
import os

# ✅ Step 1: Check if script starts
print("🚀 Script started...")

# ✅ Step 2: Check working directory
print("📂 Current working directory:", os.getcwd())

# ✅ Step 3: Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

if torch.cuda.is_available():
    print("🟢 GPU Memory Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("🟢 GPU Memory Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

# ✅ Step 4: Check dataset loading
try:
    print("📊 Checking dataset loading...")
    for batch_idx, (images, labels, target_lengths) in enumerate(train_loader):
        print(f"✅ Loaded batch {batch_idx+1}")
        break  # Stop after loading one batch (testing)
except Exception as e:
    print("❌ Dataset loading error:", e)
    exit()  # Stop execution if dataset fails

# ✅ Step 5: Initialize model, loss function, and optimizer
num_classes = 26 + 10 + 1  # 26 letters + 10 digits + 1 blank (for CTC)
model = CRNN(input_height=32, hidden_size=256, num_classes=num_classes).to(device)
criterion = nn.CTCLoss(blank=0)  # CTC requires a blank token
optimizer = optim.Adam(model.parameters(), lr=0.0000001)  # Reduce learning rate
vocab = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
char_to_idx = {char: i for i, char in enumerate(vocab)}
# ✅ Step 6: Start training
try:
    num_epochs = 50
    print("🎯 Training started...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\n📌 Epoch {epoch+1}/{num_epochs} started...")

        for batch_idx, (images, labels, target_lengths) in enumerate(train_loader):  # ✅ Expect 3 values
            print(f"🔄 Processing batch {batch_idx+1}")

            images = images.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)
            labels = labels.view(-1)  # Flatten the tensor
            print(f"🔍 Target Labels: {[vocab[i] for i in labels.tolist() if i < len(vocab)]}") 
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # Expected shape: (T, N, C)

            # Compute input lengths (assume all sequences have the same length)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.long).to(device)
            print(f"🔍 Input Lengths: {input_lengths.tolist()}")
            print(f"🔍 Target Lengths: {target_lengths.tolist()}")

            # Apply log_softmax only once
            outputs = outputs.log_softmax(2)

            # Compute CTC Loss
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            total_loss += loss.item()

            print(f"✅ Batch {batch_idx+1} processed - Loss: {loss.item()}")

        print(f"📉 Epoch {epoch+1}/{num_epochs} completed - Total Loss: {total_loss}")

    print("🏆 Training completed successfully!")

    # ✅ Step 7: Save the model
    model_path = "./handwriting_ocr.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved successfully as {model_path}")

except Exception as e:
    print("❌ Error occurred during training:", e)
