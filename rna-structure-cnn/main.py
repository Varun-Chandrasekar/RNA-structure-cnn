import argparse
from src.train import train_model
from src.predict import predict_structure, load_model

def main():
    parser = argparse.ArgumentParser(description="RNA Structure CNN")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Run mode')
    parser.add_argument('--seq', type=str, help='RNA sequence for prediction')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.seq:
            print("Please provide a sequence using --seq for prediction.")
        else:
            model = load_model()
            structure = predict_structure(model, args.seq)
            print(f"Sequence : {args.seq}")
            print(f"Structure: {structure}")

if __name__ == "__main__":
    main()