from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this will take a while first time)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print("✅ Model loaded\n")

while True:
    prompt = input("You: ")

    inputs = tokenizer(prompt, return_tensors="pt")
    print("🤖 Thinking...")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    

    print("\nAI:", tokenizer.decode(outputs[0], skip_special_tokens=True), "\n")