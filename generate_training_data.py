import json

# Load legal opinions (JSONL)
cases = []
with open("legal_opinion_chunks.json") as f:
    for line in f:
        line = line.strip()
        if line:
            cases.append(json.loads(line))

# Load judge profiles (JSON array)
with open("judge_profiles.json") as f:
    judges = json.load(f)

qa_pairs = []

# Q1: Closest precedent (for each case)
for case in cases:
    question = f"What precedent is most similar to the case of {case['case_title']}?"
    answer = f"{case['case_title']} concerns: {case['text'][:300]}..."
    qa_pairs.append({"input": question, "output": answer})

# Q2: Judge swap (for each case and judge)
for case in cases:
    for judge in judges:
        question = (
            f"If Judge {judge['Name']} had presided over {case['case_title']}, "
            "how might the decision and societal impact differ?"
        )
        judge_desc = judge.get('Judicial district', '')
        answer = (
            f"Judge {judge['Name']} is known for {judge_desc}. "
            f"Profile: {json.dumps(judge)}. "
            "Based on their background and leanings, the outcome might have differed in..."
        )
        qa_pairs.append({"input": question, "output": answer})

# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair) + "\n")
