from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch



def load_translation_dataset(tokenizer, data_path="data/nusax_mt_eng_jav_seacrowd_t2t"):
    """Load translation dataset from parquet files."""
    def preprocess(item):

        tokenized_input = tokenizer(
            item['text_1'],
            text_target = item['text_2'],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return tokenized_input  
    
    dataset = load_from_disk(data_path)
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset



def get_dataloader_splits(
    data_path="data/nusax_mt_eng_jav_seacrowd_t2t",
    max_length=128,
    batch_size=32,
    device=None,
):
    """Create train, validation, and test DataLoaders for the NusaX translation dataset.

    Args:
        data_path: Path to the saved dataset on disk
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for DataLoader

    Returns:
        dict with "train", "val", "test" DataLoaders and the tokenizer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "Helsinki-NLP/opus-mt-id-en"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset_dict = load_translation_dataset(tokenizer, data_path)


    splits = ["train", "validation", "test"]
    dataloaders = {}

    collator = DataCollatorForSeq2Seq(tokenizer)

    for disk_key in splits:
        ds = dataset_dict[disk_key]
        # ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataloaders[disk_key] = DataLoader(ds, batch_size=batch_size, shuffle=(disk_key == "train"), collate_fn=collator)

    return dataloaders, tokenizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, tokenizer = get_dataloader_splits()
    print(f"TOKENIZER: {tokenizer.bos_token_id}")
    print(f"special ids: {tokenizer.all_special_tokens} - {tokenizer.all_special_ids}")

    for name in ["train", "validation", "test"]:
        dl = dataloaders[name]
        print(f"\n{name}: {len(dl.dataset)} samples, {len(dl)} batches")
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch['input_ids'])
            print(f"Type: {batch['input_ids'].dtype}")
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Sample input: {tokenizer.decode(batch['input_ids'][0].tolist())}")
            print(f"  Sample target: {tokenizer.decode(batch['labels'][0].tolist())}")
            print(f"  Attention mask: {(batch['attention_mask'][0].tolist())}")
            break
