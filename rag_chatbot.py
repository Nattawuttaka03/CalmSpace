import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_system import retrieve

# ใช้ CPU เต็ม
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

print("CPU cores:", os.cpu_count())

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model.eval()
print("Model loaded ✅")

while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    # 🔎 Retrieve context
    context = retrieve(question)

    # 🔹 รวม context ให้เป็นข้อความเดียว
    context_text = "\n\n".join(context)

    full_prompt = f"""
You are a warm and caring third-gender AI.

CRITICAL STYLE RULES:
- NEVER use formal or bureaucratic language
- Speak like a close friend who listens and cares

Response style:
- Natural and friendly
- Short and clear
- Use only relevant context
- If the topic is serious, be gentle and supportive


Context:
{context}

Question:
{question}

Answer:
"""

    # ✅ Tokenize แล้วส่งเข้า CPU
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")
    print("Prompt tokens:", inputs["input_ids"].shape[1])

    # 🔎 Debug ความยาว prompt (อยากรู้ว่าช้าเพราะยาวไหม)
    # print("Prompt tokens:", inputs["input_ids"].shape[1])

    # ⚡ Generate เร็วขึ้นมาก
    import time

    start = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,        # 🔥 เปิด sampling
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        print("⏱ generate time:", time.time() - start)

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print("\n🤖 AI:")
    print(answer)
    print("\n" + "=" * 50)
    
