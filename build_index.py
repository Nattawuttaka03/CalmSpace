from langchain_core.documents import Document
import json
import os
import torch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

DATA_PATH = "D:\\Monor\\NLP\\Final project\\dataset"
docs = []

def load_json_safely(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

for file in os.listdir(DATA_PATH):
    if file.endswith(".json"):
        path = os.path.join(DATA_PATH, file)
        data = load_json_safely(path)

        # 🔹 intents format
        if isinstance(data, dict) and "intents" in data:
            for intent in data["intents"]:
                patterns = intent.get("patterns", [])
                responses = intent.get("responses", [])

                for p in patterns:
                    for r in responses:
                        text = f"User: {p}\nAssistant: {r}"
                        docs.append(Document(page_content=text))

        else:
            items = data if isinstance(data, list) else [data]

            for item in items:

                # input/output
                if "input" in item and "output" in item:
                    user = item["input"]
                    bot = item["output"]

                # Context/Response
                elif "Context" in item and "Response" in item:
                    user = item["Context"]
                    bot = item["Response"]

                else:
                    continue

                text = f"""The following is a supportive mental health conversation.

User: {user}
Assistant: {bot}
"""
                docs.append(Document(page_content=text))
print("SAVE PATH =", os.getcwd())
print("Total documents =", len(docs))


from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# 🔹 Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
texts = [chunk.page_content for chunk in chunks]

print("Total chunks =", len(texts))

# 🔹 Embed
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 🔹 FAISS
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# 🔹 Save ให้อยู่โฟลเดอร์เดียวกับไฟล์นี้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

faiss.write_index(index, os.path.join(BASE_DIR, "faiss_index"))

with open(os.path.join(BASE_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("✅ Index built successfully")