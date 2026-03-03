from unsloth import FastLanguageModel
import torch
from prompt_template import build_prompt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",   # รอ access
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"


def ask_llama(question):
    prompt = build_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


while True:
    q = input("You: ")
    print("AI:", ask_llama(q))