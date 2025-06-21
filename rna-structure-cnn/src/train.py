import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, DataLoader

from src.dataset import load_from_inner_folder, one_hot_encode, structure_to_labels
from src.model import RNACNN

# Load and prepare data
class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, struct = self.data[idx]
        x = torch.tensor(one_hot_encode(seq), dtype=torch.float32)
        y = torch.tensor(structure_to_labels(struct), dtype=torch.long)
        return x, y

def collate_batch(batch):
    x_batch, y_batch = zip(*batch)
    x_padded = pad_sequence(x_batch, batch_first=True)
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=-1)
    return x_padded, y_padded

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = RNADataset(train_data)
val_dataset = RNADataset(val_data)

# Initialize model, loss, optimizer
model = RNACNN()
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
val_losses = []
val_accuracies = []

# Training loop
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_batch, shuffle=True)
train_losses = []
for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        outputs = outputs.contiguous().view(-1, 3)
        y_batch = y_batch.view(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Avg Loss: {avg_loss:.4f}")

# Final evaluation
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)  # shape: (batch, seq_len, 3)
            outputs = outputs.contiguous().view(-1, 3)
            y_batch_flat = y_batch.view(-1)

            loss = criterion(outputs, y_batch_flat)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            mask = y_batch_flat != -1
            correct = (preds == y_batch_flat) & mask
            total_correct += correct.sum().item()
            total_count += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_count
    return avg_loss, accuracy

val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
val_losses.append(val_loss)
val_accuracies.append(val_accuracy)

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/rna_cnn_trained_1.pt")
print("Model weights saved to models/rna_cnn_trained_1.pt")
