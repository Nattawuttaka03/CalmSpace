from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


# โหลด embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# โหลด index
index = faiss.read_index("faiss_index")

# โหลดข้อความ
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def retrieve(query, k=5):
    # 🔹 encode query
    q_emb = model.encode(
        query,
        convert_to_numpy=True
    )

    # 🔹 search
    distances, indices = index.search(
        np.array([q_emb]),
        k
    )

    # 🔹 ดึงข้อความ
    results = [chunks[i] for i in indices[0]]

    # 🔹 ลบ duplicate
    unique_results = list(dict.fromkeys(results))

    # 🔹 ส่งกลับแค่ 3 อันที่ไม่ซ้ำ
    return unique_results[:3]

# print(retrieve("I feel so worthless"))