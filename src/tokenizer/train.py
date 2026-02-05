"""Train a SentencePiece tokenizer on the prepared training text."""
import sentencepiece as spm


def train_tokenizer(
    input_path="src/tokenizer/train.all.txt",
    model_prefix="src/tokenizer/marian/jp_en_tokenizer",
    vocab_size=32000,
):
    """Train a SentencePiece tokenizer.

    Args:
        input_path: Path to training text file
        model_prefix: Output path prefix for model files
        vocab_size: Size of the vocabulary
    """
    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.999,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
    print(f"Tokenizer trained and saved to {model_prefix}.model")


if __name__ == "__main__":
    train_tokenizer()
