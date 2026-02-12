# Project Structure — English-Japanese Machine Translation

An NLP project for English-to-Japanese (and reverse) machine translation using transformer models, with custom SentencePiece tokenization and HuggingFace infrastructure.

---

## Directory Layout

```
project/
├── requirements.txt                          # Python dependencies
├── data/                                     # Raw dataset (gitignored)
└── src/
    ├── USAGE.md                              # Pipeline usage docs
    ├── data/
    │   ├── __init__.py                       # Exports get_dataloader, tokenize_batch
    │   ├── dataset.py                        # Dataset loading, tokenization, DataLoader
    │   └── prepare_text.py                   # Extract raw text for tokenizer training
    ├── tokenizer/
    │   ├── __init__.py                       # Exports load_tokenizer()
    │   ├── train.py                          # SentencePiece tokenizer training script
    │   ├── train.all.txt                     # Raw training text (~1.8 GB)
    │   └── marian/
    │       ├── jp_en_tokenizer.model         # Trained SentencePiece model (32k vocab)
    │       ├── jp_en_tokenizer.vocab         # Vocabulary file
    │       └── tokenizer.config              # Tokenizer config (JSON)
    ├── train/
    │   └── train_translation_model.py        # Training script (placeholder)
    ├── train.py                              # Training utilities & dataset class
    ├── translation_model_eng_to_jp.py        # Pre-trained EN→JP demo
    ├── translation_model_jp_to_eng.py        # Pre-trained JP→EN demo
    ├── translation_model_eng_to_jp_untrained.py  # Untrained model initialization test
    ├── downloading_dataset_jp_eng.py         # HuggingFace dataset downloader
    └── test_llama_model.py                   # Llama 3.2 model test
```

---

## File Descriptions

### Data Module — `src/data/`

| File | Purpose |
|------|---------|
| `__init__.py` | Exports `get_dataloader` and `tokenize_batch` |
| `dataset.py` | Core data pipeline: loads parquet files, tokenizes text, builds DataLoaders |
| `prepare_text.py` | Extracts English + Japanese text from parquet into a single text file for tokenizer training |

### Tokenizer Module — `src/tokenizer/`

| File | Purpose |
|------|---------|
| `__init__.py` | Exports `load_tokenizer(model_path)` convenience function |
| `train.py` | Trains a SentencePiece tokenizer on `train.all.txt` |
| `marian/` | Directory containing the trained tokenizer model, vocab, and config |

### Translation Model Demos — `src/`

| File | Purpose |
|------|---------|
| `translation_model_eng_to_jp.py` | Runs EN→JP translation using pre-trained `Helsinki-NLP/opus-mt-en-jap` |
| `translation_model_jp_to_eng.py` | Runs JP→EN translation using pre-trained `Helsinki-NLP/opus-mt-jap-en` |
| `translation_model_eng_to_jp_untrained.py` | Initializes an untrained MarianMT model from config (random weights) |

### Training — `src/`

| File | Purpose |
|------|---------|
| `train.py` | Training utilities: dataset loading, tokenization mapping, `JPtoEngDataset` class |
| `train/train_translation_model.py` | Main training script (currently empty placeholder) |

### Utilities — `src/`

| File | Purpose |
|------|---------|
| `downloading_dataset_jp_eng.py` | Downloads `HuggingFaceFW/finetranslations` (Japanese subset) via `huggingface_hub` |
| `test_llama_model.py` | Tests `meta-llama/Llama-3.2-3B` for NLP concept generation |

---

## Universal Classes

### `JPtoEngDataset` — `src/train.py:25`

Custom PyTorch `Dataset` for translation pairs.

```python
class JPtoEngDataset(Dataset):
    def __init__(self, src_ids, tgt_ids):
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
```

**Status**: Incomplete — missing `__len__` and `__getitem__` methods.

---

## Models

### MarianMTModel (Pre-trained)

- **Library**: `transformers.MarianMTModel`
- **Checkpoints**:
  - `Helsinki-NLP/opus-mt-en-jap` — English → Japanese
  - `Helsinki-NLP/opus-mt-jap-en` — Japanese → English
- **Architecture**: Sequence-to-sequence transformer (encoder-decoder)
- **Used in**: `translation_model_eng_to_jp.py`, `translation_model_jp_to_eng.py`

### MarianMTModel (Untrained)

- **Created from**: `MarianConfig.from_pretrained("Helsinki-NLP/opus-mt-en-jap")`
- **Instantiated with**: `MarianMTModel(config)` (random weights)
- **Used in**: `translation_model_eng_to_jp_untrained.py`
- **Purpose**: Testing model architecture initialization without pre-trained weights

### Llama 3.2 (3B)

- **Library**: `transformers.AutoModelForCausalLM`
- **Checkpoint**: `meta-llama/Llama-3.2-3B`
- **Used in**: `test_llama_model.py`
- **Purpose**: NLP concept explanation (not for translation)

---

## Datasets & DataLoaders

### Dataset Source

- **Origin**: `HuggingFaceFW/finetranslations` (Japanese subset)
- **Format**: Parquet files at `data/finetranslations/data/jpn_Jpan/*.parquet`
- **Key Columns**:
  - `og_full_text` — Original English text
  - `translated_text` — Japanese translation
- **Download script**: `downloading_dataset_jp_eng.py`

### `load_translation_dataset()` — `src/data/dataset.py:7`

Loads parquet files into a HuggingFace `Dataset`, keeping only the two translation columns.

```python
def load_translation_dataset(data_path, num_files=1) -> Dataset
```

### `tokenize_batch()` — `src/data/dataset.py:19`

Tokenizes a batch of translation pairs using SentencePiece. Pads/truncates to `max_length`.

```python
def tokenize_batch(batch, tokenizer, max_length=128) -> dict
# Returns: {"input_ids": [...], "target_ids": [...]}
```

### `get_dataloader()` — `src/data/dataset.py:50`

End-to-end function: loads data, tokenizes, and returns a PyTorch DataLoader.

```python
def get_dataloader(
    tokenizer_path="./src/tokenizer/marian/jp_en_tokenizer.model",
    data_path="data/finetranslations/data/jpn_Jpan/*.parquet",
    max_length=128,
    batch_size=32,
    shuffle=True,
    subset_ratio=500   # Use 1/N of dataset; set to 1 for full
) -> tuple[DataLoader, SentencePieceProcessor]
```

**Output tensors**: `input_ids` and `target_ids`, both shape `[batch_size, max_length]`.

---

## Tokenizers

### SentencePiece (Custom-trained)

- **Model file**: `src/tokenizer/marian/jp_en_tokenizer.model`
- **Vocab size**: 32,000
- **Special tokens**: `pad=0`, `bos=1`, `eos=2`, `unk=3`
- **Character coverage**: 0.999
- **Training data**: `src/tokenizer/train.all.txt` (alternating English/Japanese lines)
- **Training script**: `src/tokenizer/train.py`
- **Load function**: `src/tokenizer/__init__.py → load_tokenizer(model_path)`

### MarianTokenizer (Pre-trained)

- **Library**: `transformers.MarianTokenizer`
- **Loaded with**: `MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-jap")`
- **Used in**: Pre-trained translation demo scripts

---

## Data Pipeline

```
1. Download          downloading_dataset_jp_eng.py
   HuggingFace Hub ──────────────────────────────► data/finetranslations/

2. Prepare Text      src/data/prepare_text.py
   Parquet files ────────────────────────────────► src/tokenizer/train.all.txt

3. Train Tokenizer   src/tokenizer/train.py
   train.all.txt ───────────────────────────────► marian/jp_en_tokenizer.model

4. Build DataLoader  src/data/dataset.py
   Parquet + Tokenizer ─────────────────────────► PyTorch DataLoader
                                                   (input_ids, target_ids)
```

---

## Default Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| `vocab_size` | 32,000 | `tokenizer/train.py` |
| `max_length` | 128 | `data/dataset.py` |
| `batch_size` | 32 | `data/dataset.py` |
| `shuffle` | True | `data/dataset.py` |
| `subset_ratio` | 500 | `data/dataset.py` |
| `character_coverage` | 0.999 | `tokenizer/train.py` |
| `tokenizer_path` | `./src/tokenizer/marian/jp_en_tokenizer.model` | `data/dataset.py` |
| `data_path` | `data/finetranslations/data/jpn_Jpan/*.parquet` | `data/dataset.py` |

---

## Known Issues

1. **Broken import**: `src/train.py:4` imports `from .custom_dataset import preprocess`, but `custom_dataset/` was deleted in the last commit. This will cause an `ImportError`.
2. **Incomplete class**: `JPtoEngDataset` in `src/train.py` only defines `__init__` — missing `__len__` and `__getitem__`.
3. **Empty training script**: `src/train/train_translation_model.py` is a placeholder with no implementation.

---

## Dependencies

Key libraries from `requirements.txt`:

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.10.0 | Deep learning framework |
| `transformers` | 5.0.0 | Model architectures (MarianMT, Llama) |
| `sentencepiece` | 0.2.1 | Subword tokenization |
| `datasets` | 4.5.0 | HuggingFace dataset loading |
| `tokenizers` | 0.22.2 | Fast tokenizer backend |
| `accelerate` | 1.12.0 | Training acceleration |
| `huggingface_hub` | — | Model/dataset downloading |
