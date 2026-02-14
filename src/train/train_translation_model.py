from src.datasets.eng_jav.nusax_mt_eng_jav import get_dataloader_splits
from transformers import AutoTokenizer, MarianConfig, MarianMTModel
import sentencepiece as spm
import torch 
from tqdm import tqdm
from src.evaluation.eval_metrics import evaluate_model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #Model Loading and Setup
    parser.add_argument("--main-model-path", type=str)
    parser.add_argument("--reverse-model-path", type=str)

    args = parser.parse_args()

    saved_model_path = "./ckpt/test_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, tokenizer = get_dataloader_splits(batch_size=8)
    train_dataloader = dataloaders["train"]
    test_dataloader = dataloaders["test"]
    val_dataloader = dataloaders["validation"]

    main_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    reverse_checkpoint = "Helsinki-NLP/opus-mt-en-id"

    if args.main_model_path is not None:
        main_model = MarianMTModel.from_pretrained(args.main_model_path)
    else:
        main_model = MarianMTModel.from_pretrained(main_checkpoint)
    if args.reverse_model_path is not None:
        reverse_model = MarianMTModel.from_pretrained(args.reverse_model_path)
    else:
        reverse_model = MarianMTModel.from_pretrained(reverse_checkpoint)


    tokenizer = AutoTokenizer.from_pretrained(main_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(reverse_checkpoint)

    main_model.to(device)
    reverse_model.to(device)

    main_model.resize_token_embeddings(len(tokenizer))

    print(f"Parameters: {main_model.num_parameters()}")
    epochs = 5

    optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-4)
    
    for epoch in range(1, epochs+1):
        main_model.train()
        print(f"Epoch: {epoch}")   

        batch_loss = 0.0
        batch_index = 0

        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {k: v.to(device) for k,v in batch.items()}
            # main_model.cpu()
            # print(f"batch: {batch}")
            # print(f"batch keys: {batch.keys()}")

            outputs = main_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
                )
            # print(batch['input_ids'][0])
            # print(batch['target_ids'][0])
            loss = outputs.loss
            batch_loss += loss

            optimizer.zero_grad()

            loss.reverse()

            torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_index+=1
        evaluate_model(main_model, tokenizer, test_dataloader, device)
        print(f"Batch loss: {batch_loss/batch_index}")

    print(f"Saving main_model...")
    model_save_path="ckpt/test_model" 
    main_model.save_pretrained("ckpt/test_model")
    print(f"Saved main_model to {model_save_path}")

    