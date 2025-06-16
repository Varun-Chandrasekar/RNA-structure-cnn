# RNA-structure-cnn

This project implements a deep learning model (convolution neural network (CNN))) to predict RNA secondary structures (dot-bracket notation) from RNA sequences.
---

## Data
This project currently accepts only .bpseq and .txt files from the data folder

---

##  Project Structure

```
rna-structure-cnn-complete/
├── src/
│   ├── train.py           # CNN training script
│   ├── predict.py         # Predicts structure from input sequence
│
├── main.py                # Unified entry point for training or prediction
├── data/                  # Sample 100-record dataset
├── images/                # Validation accuracy plot
├── notebooks/             # Training and EDA notebook
├── models/                # Placeholder for trained model
├── results.txt            # Summary of loss and accuracy
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset
- **Main training dataset**: 27,366 RNA records from bp_RNA_1m_90 (https://bprna.cgrb.oregonstate.edu/) [not included in repo due to size].
- **Sample dataset**: `data/sample_bprna_100.txt` contains 100 short RNA records for testing.
---

##  Model and how it works
A simple 1D-CNN that classifies each base in the RNA sequence as paired (1) or unpaired (0) using the dot-bracket notation as labels.

- One-hot encodes RNA sequences (A, U, G, C)
- CNN learns per-base paired vs unpaired prediction
- Output is dot-bracket notation structure
- Trained on filtered bpRNA dataset (≤100 nt)

## First Final  Evaluation Metrics 
- Total samples used: 27,366
- Final validation loss: 0.6553
- Final validation accuracy: 61.48%

##  Prediction

Use `predict.py` to predict structure for a new sequence:
```bash
python predict.py
```
