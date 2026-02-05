from datasets import load_dataset
import glob


dataset_files = glob.glob("data/finetranslations/data/jpn_Jpan/*.parquet")

limited_files = dataset_files[:1]

dataset = load_dataset("parquet",  data_files=limited_files)["train"]

print(dataset)