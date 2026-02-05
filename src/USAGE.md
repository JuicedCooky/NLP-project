# Usage Guide

## Project Structure

```
src/
├── data/
│   ├── dataset.py        # Dataset loading + tokenization + DataLoader
│   └── prepare_text.py   # Generate training text for tokenizer
├── tokenizer/
│   ├── train.py          # Train SentencePiece tokenizer
│   └── marian/           # Trained tokenizer model files
└── train.py              # Main training script
```

## Pipeline

### 1. Prepare Training Text (Optional)

Generate text file for tokenizer training from parquet dataset:

```bash
python -m src.data.prepare_text
```

### 2. Train Tokenizer (Optional)

Train a new SentencePiece tokenizer:

```bash
python -m src.tokenizer.train
```

### 3. Load Data for Training

```python
from src.data import get_dataloader

dataloader, tokenizer = get_dataloader(
    tokenizer_path="./src/tokenizer/marian/jp_en_tokenizer.model",
    batch_size=32,
    max_length=128,
    shuffle=True,
)

for batch in dataloader:
    input_ids = batch['input_ids']    # (batch_size, max_length)
    target_ids = batch['target_ids']  # (batch_size, max_length)
    # ... training loop
```

### 4. Decode Tokens

```python
from src.tokenizer import load_tokenizer

tokenizer = load_tokenizer()
text = tokenizer.decode(token_ids.tolist())
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokenizer_path` | `./src/tokenizer/marian/jp_en_tokenizer.model` | Path to tokenizer model |
| `data_path` | `data/finetranslations/data/jpn_Jpan/*.parquet` | Glob pattern for data files |
| `max_length` | 128 | Max sequence length |
| `batch_size` | 32 | Batch size |
| `subset_ratio` | 500 | Use 1/N of dataset (set to 1 for full dataset) |
