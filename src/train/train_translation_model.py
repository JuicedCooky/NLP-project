from src.data.dataset import get_dataloader, get_dataloader_splits
from transformers import MarianConfig, MarianMTModel
import sentencepiece as spm
import torch 
from tqdm import tqdm
from src.evaluation.eval_metrics import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataloader, tokenizer = get_dataloader(batch_size=8,subset_ratio=250)
dataloaders, tokenizer = get_dataloader_splits(batch_size=8,subset_ratio=250)

#Loading config file to train translation model
config = MarianConfig(
    vocab_size=32000,
    d_model=512,             
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    pad_token_id=tokenizer.pad_id(),
    eos_token_id=tokenizer.eos_id(),
    bos_token_id=tokenizer.bos_id(),
)

model = MarianMTModel(config)
model.to(device)

model.resize_token_embeddings(len(tokenizer))

print(f"Parameters: {model.num_parameters()}")
epochs = 4

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.config.pad_token_id = 0
model.config.decoder_start_token_id = 1
model.config.eos_token_id = 2
model.config.unk_token_id = 2

model.generation_config.pad_token_id = 0
model.generation_config.decoder_start_token_id = 1
model.generation_config.eos_token_id = 2
model.generation_config.unk_token_id = 2

for epoch in range(1, epochs+1):
    model.train()
    print(f"Epoch: {epoch}")   

    batch_loss = 0.0
    batch_index = 0

    loop = tqdm(dataloaders["train"], leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k,v in batch.items()}
        # model.cpu()

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['target_ids']
            )
        # print(batch['input_ids'][0])
        # print(batch['target_ids'][0])
        loss = outputs.loss
        batch_loss += loss

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_index+=1
    evaluate_model(model, tokenizer, dataloaders['test'], device)
    print(f"Batch loss: {batch_loss/batch_index}")

print(f"Saving model...")
model_save_path="ckpt/test_model" 
model.save_pretrained("ckpt/test_model")
print(f"Saved model to {model_save_path}")

    