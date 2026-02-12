from src.data.dataset import get_dataloader
from transformers import MarianConfig, MarianMTModel
import sentencepiece as spm
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader, tokenizer = get_dataloader()

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

print(f"Parameters: {model.num_parameters()}")

epochs = 1

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(1, epochs+1):
    print(f"epoch:{epoch}")   
    for batch_idx, batch in enumerate(dataloader):
        # input_batch = {k: v.to(device) for k,v in batch['input_ids'].item()}
        print(f"input_ids: {batch['input_ids']}")
        print(f"target_ids: {batch['target_ids']}")
        print(f"attention_mask: {batch['attention_mask']}")
        # print(batch.keys())
    