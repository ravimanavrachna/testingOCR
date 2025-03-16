import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_height, hidden_size, num_classes):
        super(CRNN, self).__init__()

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # Reduces height & width by half
        )

        # Calculate LSTM input size
        self.lstm_input_size = 64 * (input_height // 2)  # Height reduced by MaxPool2d

        # BiLSTM for sequence modeling
        self.rnn = nn.LSTM(self.lstm_input_size, hidden_size, bidirectional=True, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # BiLSTM doubles hidden size

    def forward(self, x):
        batch_size = x.size(0)

        # Pass through CNN
        x = self.conv(x)  # Shape: (batch, 64, new_height, width)

        # Reshape for LSTM
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 2, 1)  # (batch, width, height, channels)
        x = x.reshape(batch_size, width, -1)  # (batch, width, features)

        # **Ensure LSTM input size is correct**
        assert x.shape[2] == self.lstm_input_size, f"Expected {self.lstm_input_size}, got {x.shape[2]}"

        # Pass through LSTM
        x, _ = self.rnn(x)  # Shape: (batch, width, hidden_size * 2)

        # Fully connected output
        x = self.fc(x)  # Shape: (batch, width, num_classes)

        return x.permute(1, 0, 2).log_softmax(2)  # Change to (T, N, C) for CTC loss

# Initialize model
num_classes = 26 + 10 + 1  # 26 letters + 10 digits + 1 blank label for CTC
model = CRNN(input_height=32, hidden_size=256, num_classes=num_classes)
