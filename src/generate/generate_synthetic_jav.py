import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--forward-model-path", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backward_checkpoint = "Helsinki-NLP/opus-mt-en-id"

    forward_model = 