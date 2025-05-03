import json
import os
import numpy as np
from together import Together
import faiss

# 1. Setup TogetherAI client
client = Together(api_key="tgp_v1_4Dgcnv-_rmTV4EIDl48zUpT2FmtcCHV3aGOtm2R9mwU")

# 2. Helper to get embeddings from TogetherAI
def get_embedding(text):
    resp = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",  # Or another Together embedding model
        input=[text]
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# 3. Load and embed legal opinions
cases = []
case_embeddings = []
with open("legal_opinion_chunks.json") as f:
    for line in f:
        case = json.loads(line)
        cases.append(case)
        case_embeddings.append(get_embedding(case['text']))

case_embeddings = np.vstack(case_embeddings)

# 4. Load and embed judge profiles
judges = []
judge_embeddings = []
with open("judge_profiles.json") as f:
    judges = json.load(f)
    for judge in judges:
        profile_text = json.dumps(judge)
        judge_embeddings.append(get_embedding(profile_text))
judge_embeddings = np.vstack(judge_embeddings)

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