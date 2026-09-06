from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import chromadb
import requests

app = FastAPI()

# 允许WordPress网页跨域请求
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("company_docs")


HF_TOKEN = os.environ.get("HF_TOKEN")
#HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# ⬅️ 新增：改用API算向量的函数
hf_client = InferenceClient(token=HF_TOKEN)

def get_embedding(text):
    result = hf_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
    if hasattr(result, 'ndim') and result.ndim == 2:
        result = result.mean(axis=0)
    return result.tolist()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    # 1. 检索相关文档 —— 这里换掉了
    query_embedding = get_embedding(q.question)  # ⬅️ 原本是 model.encode([q.question]).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)  # ⬅️ 外面加 [ ]
    context = "\n".join(results['documents'][0])

    prompt = f"根据以下紫微斗数文件回答问题，若文件没提到就说不知道。\n文件内容：{context}\n问题：{q.question}\n回答："
    
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    answer = response.json()["choices"][0]["message"]["content"]
    return {"answer": answer}