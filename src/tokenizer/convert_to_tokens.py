from datasets import load_dataset
import glob
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import torch

dataset_files = glob.glob("data/finetranslations/data/jpn_Jpan/*.parquet")
limited_files = dataset_files[:1]

dataset = load_dataset("parquet",  data_files=limited_files, split="train")
keys_to_remove = [key for key in dataset[0].keys() if key!="og_full_text" and key!="translated_text"]
dataset = dataset.remove_columns(keys_to_remove)

print(len(dataset)//500)


tokenizer = spm.SentencePieceProcessor(model_file="./src/tokenizer/marian/jp_en_tokenizer.model")

max_length = 128
pad_id = tokenizer.pad_id()

def tokenize_batch(batch):
    input_list = []
    output_list = []
    print(len(batch))
    print(len(batch["og_full_text"]))
    for i in range(len(batch["og_full_text"])):
        input = tokenizer.encode(batch["og_full_text"][i])
        output = tokenizer.encode(batch["translated_text"][i])
        concat_list = [input,output] 
        for i in range(len(concat_list)):
            if len(concat_list[i]) < max_length:
                concat_list[i] = (concat_list[i] + [pad_id] * (max_length - len(concat_list[i])))
            elif len(concat_list[i]) > max_length:
                concat_list[i] = concat_list[i][:max_length]
        input_list.append(concat_list[0])
        output_list.append(concat_list[1])

    return {
        "input_ids": (input_list), 
        "target_ids": (output_list), 
    }


dataset = dataset.select(range(len(dataset)//500))
dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["og_full_text", "translated_text"])
dataset.set_format(type="torch", columns=['input_ids','target_ids'])

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

count = 0
for dict in (dataloader):
    # input = dict[0]
    # target = dict[1]
    print(dict['input_ids'].shape)
    count += 1
    if count == 1:
        print(tokenizer.decode(dict['input_ids'].tolist()))
        print(tokenizer.decode(dict['target_ids'].tolist()))
# print(f"count:{count}")
# dataset.save_to_disk("src/tokenizer/train.all.tokenized.txt")