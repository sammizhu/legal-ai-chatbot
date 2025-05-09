import json
import os
import numpy as np
import pickle
from together import Together
import faiss

# 1. Setup TogetherAI client
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise RuntimeError("TOGETHER_API_KEY environment variable not set!")
client = Together(api_key=api_key)

# 2. Helper to get embeddings from TogetherAI
def get_embedding(text):
    resp = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        input=[text] 
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# 3. Always load precomputed embeddings and data
# Load cases
with open("legal_opinion_chunks.json") as f:
    cases = [json.loads(line) for line in f if line.strip()]
# Load case embeddings
case_embeddings = np.load("case_embeddings.npy")

# Load judges
with open("judge_profiles.json") as f:
    judges = json.load(f)
# Load judge embeddings
judge_embeddings = np.load("judge_embeddings.npy")


# 5. Build FAISS indices
case_index = faiss.IndexFlatL2(case_embeddings.shape[1])
case_index.add(case_embeddings)
judge_index = faiss.IndexFlatL2(judge_embeddings.shape[1])
judge_index.add(judge_embeddings)

# 6. Retrieval + Generation
def answer_query(query, top_k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    # Retrieve cases
    _, case_idxs = case_index.search(query_emb, top_k)
    retrieved_cases = [cases[i] for i in case_idxs[0]]
    # Retrieve judges
    _, judge_idxs = judge_index.search(query_emb, top_k)
    retrieved_judges = [judges[i] for i in judge_idxs[0]]

    # Prepare context
    context = ""
    context += "Relevant Cases:\n"
    for c in retrieved_cases:
        context += f"- {c['case_title']} ({c['court']}, {c['date']}): {c['text'][:300]}...\n"
    context += "\nRelevant Judges:\n"
    for j in retrieved_judges:
        context += f"- {j['Name']}: {j.get('Judicial district', '')}, Leanings: {j.get('x_dem', '')}, {j.get('x_repub', '')}\n"

    # Generate answer with TogetherAI LLM
    prompt = f"{context}\n\nUser Question: {query}\n\nAnswer:"
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    user_query = input("Enter your legal question: ")
    answer = answer_query(user_query)
    print("\nAI Answer:\n", answer)