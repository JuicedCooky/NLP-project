import sentencepiece as spm
from datasets import load_dataset
import glob

# spm.SentencePieceTrainer.Train()

dataset_files = glob.glob("finetranslations/data/jpn_Jpan/*.parquet")
limited_files = dataset_files[:1]


dataset = load_dataset("parquet", data_files=limited_files)["train"]
eng = dataset["og_full_text"] 
jp = dataset["translated_text"] 

with open("tokenizer/train.all.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(eng[i].replace("\n", " ") + "\n")
        f.write(jp[i].replace("\n", " ") + "\n")