from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import requests

app = FastAPI()

# 允许WordPress网页跨域请求
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("company_docs")

import os
HF_TOKEN = os.environ.get("HF_TOKEN")
#HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    # 1. 检索相关文档
    query_embedding = model.encode([q.question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    context = "\n".join(results['documents'][0])

    # 2. 组织提示词，调用Hugging Face免费模型
    prompt = f"根据以下紫微斗数文件回答问题，若文件没提到就说不知道。\n文件内容：{context}\n问题：{q.question}\n回答："
    
    #response = requests.post(
    #    HF_API_URL,
    #    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    #    json={"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    #)
    #answer = response.json()[0]['generated_text']
    
    response = requests.post(
    HF_API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
)
    answer = response.json()["choices"][0]["message"]["content"]
    return {"answer": answer}