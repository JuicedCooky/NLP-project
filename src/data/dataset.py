from datasets import load_dataset
import glob
import sentencepiece as spm
from torch.utils.data import DataLoader


def load_translation_dataset(data_path="data/finetranslations/data/jpn_Jpan/*.parquet", num_files=1, split="train"):
    """Load translation dataset from parquet files."""
    dataset_files = glob.glob(data_path)
    limited_files = dataset_files[:num_files]

    dataset = load_dataset("parquet", data_files=limited_files, split=split)
    keys_to_remove = [key for key in dataset[0].keys() if key != "og_full_text" and key != "translated_text"]
    dataset = dataset.remove_columns(keys_to_remove)

    return dataset


def tokenize_batch(batch, tokenizer, max_length=128):
    """Tokenize a batch of translation pairs."""
    pad_id = tokenizer.pad_id()
    input_list = []
    output_list = []
    attention_mask_list = []

    for i in range(len(batch["og_full_text"])):
        input_tokens = tokenizer.encode(batch["og_full_text"][i])
        output_tokens = tokenizer.encode(batch["translated_text"][i])
        mask = []
        # Pad or truncate input
        
        input_len = len(input_tokens)

        if input_len < max_length:
            mask = [1] * input_len + [0] * (max_length - input_len)
            input_tokens = input_tokens + [pad_id] * (max_length - input_len)
        else:
            mask = [1] * max_length
            input_tokens = input_tokens[:max_length]

        # Pad or truncate output
        if len(output_tokens) < max_length:
            output_tokens = output_tokens + [pad_id] * (max_length - len(output_tokens))
        else:
            output_tokens = output_tokens[:max_length]

        input_list.append(input_tokens)
        output_list.append(output_tokens)
        attention_mask_list.append(mask)
        

    return {
        "input_ids": input_list,
        "target_ids": output_list,
        "attention_mask": attention_mask_list
    }


def get_dataloader(
    tokenizer_path="./src/tokenizer/marian/jp_en_tokenizer.model",
    data_path="data/finetranslations/data/jpn_Jpan/*.parquet",
    max_length=128,
    batch_size=32,
    shuffle=True,
    subset_ratio=500,
    split="train",
):
    """Create a DataLoader for the translation dataset.

    Args:
        tokenizer_path: Path to the SentencePiece model file
        data_path: Glob pattern for parquet data files
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        subset_ratio: Use 1/subset_ratio of the dataset (for testing)

    Returns:
        DataLoader and tokenizer
    """
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    dataset = load_translation_dataset(data_path, split=split)

    # Use subset for faster iteration
    if subset_ratio > 1:
        dataset = dataset.select(range(len(dataset) // subset_ratio))

    # Tokenize
    dataset = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["og_full_text", "translated_text"],
        load_from_cache_file=False
    )
    dataset.set_format(type="torch", columns=['input_ids', 'target_ids', 'attention_mask'])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, tokenizer


if __name__ == "__main__":
    # Test the dataloader
    dataloader, tokenizer = get_dataloader()

    for batch in dataloader:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Target shape: {batch['target_ids'].shape}")
        print(f"Sample input: {tokenizer.decode(batch['input_ids'][0].tolist())}")
        print(f"Sample target: {tokenizer.decode(batch['target_ids'][0].tolist())}")
        break
