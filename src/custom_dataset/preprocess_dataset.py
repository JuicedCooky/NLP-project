from datasets import load_dataset
import glob
import sentencepiece as spm
import os

def preprocess (tokenizer, batch):
    inputs = tokenizer.encode(batch["og_full_text"], truncation=True, padding="max_length",max_length=max_length)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["translated_text"], truncation=True, padding="max_length",max_length=max_length)
    
    inputs["labels"] = labels["input_ids"]
    
    return inputs
