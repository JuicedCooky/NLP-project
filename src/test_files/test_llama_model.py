from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain backtranslation in NLP."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    **inputs, 
    max_new_tokens=200,
    do_sample = True,
    top_p = 0.9,
    repetition_penalty = 1.2,
    eos_token_id =  tokenizer.eos_token_id,
    temperature = 0.7
    )
print(tokenizer.decode(output[0]))
