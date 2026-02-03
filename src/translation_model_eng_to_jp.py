from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-jap"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Example source text (English)
source_texts = ["Hello, how are you?", "Good morning!", "What is your name?"]

# Tokenize the input texts
inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)

# Generate translations
translated_ids = model.generate(inputs["input_ids"])

# Decode the generated tokens to get the translated text
translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

# Print the translations
for src, tgt in zip(source_texts, translated_texts):
    print(f"Source: {src} => Translated: {tgt}")
