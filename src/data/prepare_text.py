"""Prepare training text file for SentencePiece tokenizer training."""
from datasets import load_dataset
import glob


def prepare_training_text(
    data_path="data/finetranslations/data/jpn_Jpan/*.parquet",
    output_path="src/tokenizer/train.all.txt",
    num_files=1,
):
    """Extract text from parquet files and write to a text file for tokenizer training.

    Args:
        data_path: Glob pattern for parquet data files
        output_path: Path to write the training text
        num_files: Number of parquet files to use
    """
    dataset_files = glob.glob(data_path)
    limited_files = dataset_files[:num_files]

    dataset = load_dataset("parquet", data_files=limited_files)["train"]
    eng = dataset["og_full_text"]
    jp = dataset["translated_text"]

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            f.write(eng[i].replace("\n", " ") + "\n")
            f.write(jp[i].replace("\n", " ") + "\n")

    print(f"Wrote {len(dataset) * 2} lines to {output_path}")


if __name__ == "__main__":
    prepare_training_text()
