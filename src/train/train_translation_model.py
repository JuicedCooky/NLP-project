from src.data.eng_jav.nusax_mt_eng_jav import get_dataloader_splits
from transformers import AutoTokenizer, MarianConfig, MarianMTModel
import sentencepiece as spm
import torch 
from tqdm import tqdm
from src.evaluation.eval_metrics import evaluate_model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Model Loading and Setup
    parser.add_argument("--forward-model-path", type=str)
    parser.add_argument("--backward-model-path", type=str)

    args = parser.parse_args()

    saved_model_path = "./ckpt/test_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, tokenizer = get_dataloader_splits(batch_size=8)
    train_dataloader = dataloaders["train"]
    test_dataloader = dataloaders["test"]
    val_dataloader = dataloaders["validation"]

    forward_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    backward_checkpoint = "Helsinki-NLP/opus-mt-en-id"

    if args.forward_model_path is not None:
        forward_model = MarianMTModel.from_pretrained(args.forward_model_path)
    else:
        forward_model = MarianMTModel.from_pretrained(forward_checkpoint)
    if args.backward_model_path is not None:
        backward_model = MarianMTModel.from_pretrained(args.backward_model_path)
    else:
        backward_model = MarianMTModel.from_pretrained(backward_checkpoint)


    tokenizer = AutoTokenizer.from_pretrained(forward_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(backward_checkpoint)

    forward_model.to(device)
    backward_model.to(device)

    forward_model.resize_token_embeddings(len(tokenizer))

    print(f"Parameters: {forward_model.num_parameters()}")
    epochs = 5

    optimizer = torch.optim.Adam(forward_model.parameters(), lr=1e-4)
    
    for epoch in range(1, epochs+1):
        forward_model.train()
        print(f"Epoch: {epoch}")   

        batch_loss = 0.0
        batch_index = 0

        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {k: v.to(device) for k,v in batch.items()}
            # forward_model.cpu()
            # print(f"batch: {batch}")
            # print(f"batch keys: {batch.keys()}")

            outputs = forward_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )
            # print(batch['input_ids'][0])
            # print(batch['target_ids'][0])
            loss = outputs.loss
            batch_loss += loss

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_index+=1
        evaluate_model(forward_model, tokenizer, test_dataloader, device)
        print(f"Batch loss: {batch_loss/batch_index}")

    print(f"Saving forward_model...")
    model_save_path="ckpt/test_model" 
    forward_model.save_pretrained("ckpt/test_model")
    print(f"Saved forward_model to {model_save_path}")

    