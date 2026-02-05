from datasets import load_dataset
import glob
import sentencepiece as spm
from .custom_dataset import preprocess
from torch.utils.data import DataLoader, Dataset

dataset_files = glob.glob("data/finetranslations/data/jpn_Jpan/*.parquet")
limited_files = dataset_files[:1]

dataset = load_dataset("parquet",  data_files=limited_files, split="train")


tokenizer = spm.SentencePieceProcessor(model_file="./src/tokenizer/marian/jp_en_tokenizer.model")

def tokenize_batch(batch):
    return {
        "input_ids": [tokenizer.encode(item) for item in dataset["og_full_text"]], 
        "target_ids": [tokenizer.encode(item) for item in dataset["translated_text"]], 
    }

# max_length=128
dataset = dataset.select(range(len(dataset)//10))
dataset = dataset.map(tokenize_batch, batched=True)

class JPtoEngDataset(Dataset):
    def __init__(self, src_ids, tgt_ids):
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

# for batch in dataloader:
#     print(batch["translated_text"])