# Training loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os

from src.dataset import load_from_inner_folder, one_hot_encode, structure_to_labels
from src.model import RNACNN

# Load and prepare data
data = load_from_inner_folder("data", name_filter="sample")

x_tensors = [torch.tensor(one_hot_encode(seq), dtype=torch.float32) for seq, _ in data]
y_tensors = [torch.tensor(structure_to_labels(struct), dtype=torch.long) for _, struct in data]

# Train-validation split
x_train_list, x_val_list, y_train_list, y_val_list = train_test_split(
    x_tensors, y_tensors, test_size=0.2, random_state=42
)

x_train = pad_sequence(x_train_list, batch_first=True)
y_train = pad_sequence(y_train_list, batch_first=True, padding_value=-1)
x_val = pad_sequence(x_val_list, batch_first=True)
y_val = pad_sequence(y_val_list, batch_first=True, padding_value=-1)

# Model
model = RNACNN()
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_accuracies = []

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs.view(-1, 2), y_train.view(-1))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    val_outputs = model(x_val)
    val_loss = criterion(val_outputs.view(-1, 2), y_val.view(-1))
    predicted_labels = torch.argmax(val_outputs, dim=-1)
    mask = y_val != -1
    correct = (predicted_labels == y_val) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    val_accuracies.append(accuracy)
    print(f"Final Evaluation -> Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/rna_cnn_trained_1.pt")
print("Model weights saved to models/rna_cnn_trained_1.pt")