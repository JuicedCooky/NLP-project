import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="test/tokenizer/train.all.txt",
    model_prefix = "test/tokenizer/jp_en_tokenizer",
    vocab_size = 32000,
    character_coverage=0.999,        
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    max_sentence_length=10000
)