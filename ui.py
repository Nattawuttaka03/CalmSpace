import gradio as gr
from rag_system import retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

torch.set_num_threads(torch.get_num_threads())

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model.eval()

def generate_answer(message, history):

    msg = message.lower().strip()

    
    # 🔹 simple intercept
    if re.match(r"^(hi|hello|hey)\b", msg):
        return "Hello! How can I help you?"

    emotional_keywords = ["feel", "hate", "sad", "worthless", "tired", "lonely", "angry"]

    system_prompt = (
    "You are a calm and emotionally aware assistant. "
    "When someone shares a feeling, respond with empathy and reflection. "
    "Do not give structured advice or numbered steps. "
    "Keep responses short, natural, and conversational."
)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cpu")

  
    
    # 🔥 คุมความยาวให้ 1.5B ไม่ lecture
    if any(word in msg for word in emotional_keywords):
        max_tokens = 60
    else:
        max_tokens = 50

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.45,     # 🔥 ลด randomness
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # 🔥 cleanup layer เบา ๆ (กัน header ghost)
    answer = re.sub(r"^(User:|Assistant:)\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^\d+\.\s*", "", answer)  # กัน 1. 2. 3.
    answer = re.sub(r"\n?\d+\.\s.*", "", answer)
    answer = answer.strip()

    return answer

    # --------- Custom Theme ----------
# -------- THEME -------- #
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="green"
).set(
    body_background_fill="#F0FDF4",          # พื้นหลังเขียวอ่อนมาก
    block_background_fill="#FFFFFF",
    button_primary_background_fill="#34D399",  # ปุ่มสีเขียว
    button_primary_background_fill_hover="#10B981",
)

# -------- CUSTOM CSS -------- #
css = """
/* เอา max-width ออก */
.gradio-container {
    max-width: 100% !important;
    padding-left: 10%;
    padding-right: 10%;
}

/* ให้ chat กว้างขึ้น */
.gr-chatbot {
    height: 70vh !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

/* ปรับขนาดตัวหนังสือให้สมดุล */
body {
    font-size: 15px !important;
}

.message {
    font-size: 15px !important;
}

/* bubble */
.message.user {
    background: #A7F3D0 !important;
    border-radius: 18px !important;
}

.message.bot {
    background: #DCFCE7 !important;
    border-radius: 18px !important;
}

/* input */
textarea {
    font-size: 15px !important;
    border-radius: 16px !important;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:

    gr.Markdown("""
    # 🌿 CalmSpace   
    A gentle mental health conversational assistant  
    You are safe here. Take your time.
    """)

    gr.ChatInterface(
        fn=generate_answer,
        flagging_mode="never",
        chatbot=gr.Chatbot(height="75vh")
    )

demo.launch(share=True)