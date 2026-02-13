import evaluate
import torch 
from tqdm import tqdm
from transformers import MarianMTModel, AutoTokenizer
import argparse

metric = evaluate.load("sacrebleu")

def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()

    preds = []
    refs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader):

            
            generated = model.generate(
                input_ids = batch['input_ids'].to(device),
                attention_mask = batch['attention_mask'].to(device),
                max_new_tokens=128, 
                num_beams=4,
                repetition_penalty=2.5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_lists = generated.tolist()

            decoded = [tokenizer.decode(ids) for ids in generated_lists]
            labels = [[tokenizer.decode(ids)] for ids in batch["labels"].tolist()]

            preds.extend(decoded)
            refs.extend(labels)

    print(f"PRED: {preds[0]}")
    print(f"REF:  {refs[0][0]}")
    print("-" * 20)

    result = metric.compute(predictions=preds, references=refs)
    print(f"BLEU Score: {result['score']:.2f}")

def test_translation(model, tokenizer, string="This is a test sentence.", device=None):
    # mask  = [1] * len(string) + [0] * (tokenizer.model_max_length-len(string))
    tokenized_string = tokenizer(string, padding=True, return_tensors="pt", truncation=True).to(device)

    # tokenized_string = tokenized_string + [tokenizer.pad_token_id] * (tokenizer.model_max_length-len(tokenized_string))

    output = model.generate(
        **tokenized_string,
        # attention_mask = torch.tensor(mask).unsqueeze(0).to(device),
        # decoder_start_token_id=model.config.eos_token_id
    )
    print(f"Input: {string}")
    print(f"Output: {tokenizer.decode(output, skip_special_tokens=True)}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    # saved_model_path = "./ckpt/test_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forward_checkpoint = "Helsinki-NLP/opus-mt-id-en"
    back_checkpoint = "Helsinki-NLP/opus-mt-en-id"

    if args.model_path is not None:
        model = MarianMTModel.from_pretrained(args.model_path)
    else:
        model = MarianMTModel.from_pretrained(back_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(back_checkpoint)
    model.to(device)

    test_translation(model, tokenizer, device=device)