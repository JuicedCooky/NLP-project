import evaluate
import torch 
from tqdm import tqdm

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
            labels = [[tokenizer.decode(ids)] for ids in batch["target_ids"].tolist()]

            preds.extend(decoded)
            refs.extend(labels)

    print(f"PRED: {preds[0]}")
    print(f"REF:  {refs[0][0]}")
    print("-" * 20)

    result = metric.compute(predictions=preds, references=refs)
    print(f"BLEU Score: {result['score']:.2f}")