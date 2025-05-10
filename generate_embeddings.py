import json
import numpy as np
from together import Together
import glob
import os

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

# --- Generate case embeddings ---
print("Loading cases...")
cases = []
with open("data/legal_opinion_chunks.json") as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

print(f"Generating embeddings for {len(cases)} cases...")
case_embeddings = []
for case in cases:
    case_embeddings.append(get_embedding(case['text']))
case_embeddings = np.vstack(case_embeddings)
np.save("data/case_embeddings.npy", case_embeddings)
print("Saved case_embeddings.npy")

# --- Generate judge embeddings ---
print("Loading judges...")
with open("data/judge_profiles.json") as f:
    judges = json.load(f)

print(f"Generating embeddings for {len(judges)} judges...")
judge_embeddings = []
for judge in judges:
    profile_text = json.dumps(judge)
    judge_embeddings.append(get_embedding(profile_text))
judge_embeddings = np.vstack(judge_embeddings)
np.save("data/judge_embeddings.npy", judge_embeddings)
print("Saved judge_embeddings.npy")

reassignment_jsons = [
    "data/_DistrictandOther.json",
    "data/_Fromnoncircuittousca.json",
    "data/_Fromuscctousca.json",
    "data/_reassignedFromcircuittoCircuit.json",
    "data/_ReassignedIntheFirstTerm.json",
]

for json_path in reassignment_jsons:
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        records = json.load(f)
    print(f"Generating embeddings for {len(records)} records in {json_path}...")
    embeddings = []
    for record in records:
        record_text = json.dumps(record)
        embeddings.append(get_embedding(record_text))
    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(json_path.replace(".json", "_embeddings.npy"), embeddings)
        print(f"Saved {json_path.replace('.json', '_embeddings.npy')} ({len(records)} embeddings)")
    else:
        print(f"No records found in {json_path}, skipping embedding save.")

print("All embeddings generated and saved.")
