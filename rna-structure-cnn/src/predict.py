import torch
from torch.nn.utils.rnn import pad_sequence
from src.model import RNACNN
from src.dataset import one_hot_encode

def load_model(model_path='models/rna_cnn_trained.pt'):
    model = RNACNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def decode_structure(predicted):
    mapping = {0: '.', 1: '(', 2: ')'}
    return ''.join([mapping.get(int(p), '.') for p in predicted])

def predict_structure(model, rna_seq):
    x_tensor = torch.tensor(one_hot_encode(rna_seq), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x_tensor)
        predicted = torch.argmax(output, dim=-1).squeeze(0)
        structure = decode_structure(predicted[:len(rna_seq)])
    return structure

if __name__ == "__main__":
    rna_seq = input("Enter RNA sequence (A/U/G/C only): ").strip().upper()
    if not set(rna_seq).issubset({'A', 'U', 'G', 'C'}):
        print("Invalid input: Only characters A, U, G, and C are allowed.")
    else:
        model = load_model()
        predicted_structure = predict_structure(model, rna_seq)
        print(f"Predicted structure: {predicted_structure}")
