import json
import os
import numpy as np
import pickle
from together import Together
import faiss

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise RuntimeError("TOGETHER_API_KEY environment variable not set!")
client = Together(api_key=api_key)

# Helper to get embeddings from TogetherAI
def get_embedding(text):
    resp = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        input=[text] 
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# Load legal opinion cases
with open("data/legal_opinion_chunks.json") as f:
    cases = [json.loads(line) for line in f if line.strip()]

# Load precomputed case embeddings
case_embeddings = np.load("data/case_embeddings.npy")

# Load judge profiles
with open("data/judge_profiles.json") as f:
    judges = json.load(f)

# Load precomputed judge embeddings
judge_embeddings = np.load("data/judge_embeddings.npy")

# Load judge reassignment datasets and their embeddings
reassignment_files = [
    "data/_DistrictandOther.json",
    "data/_Fromnoncircuittousca.json",
    "data/_Fromuscctousca.json",
    "data/_reassignedFromcircuittoCircuit.json",
    "data/_ReassignedIntheFirstTerm.json",
]
reassignment_jsons = []
reassignment_embeddings = []
for json_path in reassignment_files:
    with open(json_path) as f:
        data = json.load(f)
        reassignment_jsons.append(data)
    emb_path = json_path.replace(".json", "_embeddings.npy")
    emb = np.load(emb_path)
    reassignment_embeddings.append(emb)

# Build FAISS indices for fast similarity search
case_index = faiss.IndexFlatL2(case_embeddings.shape[1])
case_index.add(case_embeddings)
judge_index = faiss.IndexFlatL2(judge_embeddings.shape[1])
judge_index.add(judge_embeddings)

# Retrieval + Generation
def answer_query(query, top_k=3):
    """
    Given a user query, retrieve the top-k most relevant cases and judges using vector search,
    construct a context prompt, and generate an answer using TogetherAI LLM.
    """
    # Get embedding for the user query
    query_emb = get_embedding(query).reshape(1, -1)
    # Retrieve top-k relevant cases
    D, I = case_index.search(query_emb, top_k)
    retrieved_cases = [cases[i] for i in I[0]]
    # Retrieve top-k relevant judges
    D_j, I_j = judge_index.search(query_emb, top_k)
    retrieved_judges = [judges[i] for i in I_j[0]]

    # Build context string for the LLM prompt
    context = "Relevant Cases:\n"
    for c in retrieved_cases:
        context += f"- {c['case_title']} ({c['court']}, {c['date']}): {c['text'][:300]}...\n"
    context += "\nRelevant Judges:\n"
    def get_judge_display_name(j):
        if 'songername' in j and j['songername']:
            return j['songername']
        parts = [j.get('judgefirstname', ''), j.get('judgemiddlename', ''), j.get('judgelastname', '')]
        return ' '.join([p for p in parts if p]).strip()
    for j in retrieved_judges:
        name = get_judge_display_name(j)
        context += f"- {name}: {j.get('Judicial district', '')}, Leanings: {j.get('x_dem', '')}, {j.get('x_repub', '')}\n"

    # Generate answer with TogetherAI LLM
    prompt = f"{context}\n\nUser Question: {query}\n\nAnswer:"
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    user_query = input("Enter your legal question: ")
    answer = answer_query(user_query)
    print("\nAI Answer:\n", answer)