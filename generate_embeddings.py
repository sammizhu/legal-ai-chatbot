import json
import numpy as np
from together import Together

# Setup TogetherAI client
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

# --- Generate case embeddings ---
print("Loading cases...")
cases = []
with open("legal_opinion_chunks.json") as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

print(f"Generating embeddings for {len(cases)} cases...")
case_embeddings = []
for case in cases:
    case_embeddings.append(get_embedding(case['text']))
case_embeddings = np.vstack(case_embeddings)
np.save("case_embeddings.npy", case_embeddings)
print("Saved case_embeddings.npy")

# --- Generate judge embeddings ---
print("Loading judges...")
with open("judge_profiles.json") as f:
    judges = json.load(f)

print(f"Generating embeddings for {len(judges)} judges...")
judge_embeddings = []
for judge in judges:
    profile_text = json.dumps(judge)
    judge_embeddings.append(get_embedding(profile_text))
judge_embeddings = np.vstack(judge_embeddings)
np.save("judge_embeddings.npy", judge_embeddings)
print("Saved judge_embeddings.npy")

print("All embeddings generated and saved.")
