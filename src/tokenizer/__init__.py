import sentencepiece as spm


def load_tokenizer(model_path="./src/tokenizer/marian/jp_en_tokenizer.model"):
    """Load a trained SentencePiece tokenizer."""
    return spm.SentencePieceProcessor(model_file=model_path)
