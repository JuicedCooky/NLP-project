from src.data.dataset import get_dataloader

dataloader, tokenizer = get_dataloader()

epochs = 1

for epoch in range(1, epochs+1):
    print(f"epoch:{epoch}")   
    