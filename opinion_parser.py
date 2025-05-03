import re
from typing import List, Dict
from uuid import uuid4
import os
import json

def process_legal_opinion(filename, text: str) -> List[Dict]:
    """
    Process a structured legal opinion text (with court, title, and date headers)
    into metadata + chunked body paragraphs for embedding or storage.
    """

    # Extract header metadata
    court_match = re.search(r"Court:\s*(.*)", text)
    title_match = re.search(r"Title:\s*(.*)", text)
    date_match = re.search(r"Date:\s*(.*)", text)
    judge_match = re.search(r"Judge:\s*(.*)", text)

    court = court_match.group(1).strip() if court_match else "Unknown Court"
    title = title_match.group(1).strip() if title_match else "Unknown Case"
    date = date_match.group(1).strip() if date_match else "Unknown Date"
    judge = judge_match.group(1).strip() if judge_match else "Unknown Judge"

    plantiff, defendant = title.split(" v. ") if title_match else ("Unknown Plantiff", "Unknown Defendant")

    # Strip off headers to get the body
    body_start = text.find("\n[*") if "\n[*" in text else max(
        text.find("Date:"), text.find("Title:"), text.find("Court:"))
    body = text[body_start:].strip()

    chunk = {
        "id": filename,
        "judge": judge,
        "plantiff": plantiff,
        "defendant": defendant,
        "case_title": title,
        "court": court,
        "date": date,
        "text": body,
    }
    return chunk

"""
Process all legal opinion files in a directory.
Returns a list of dicts with chunked text and metadata for each file.
"""
all_chunks = []
directory = "judge_opinions"
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = process_legal_opinion(filename, text)
            all_chunks.append(chunks)

# Save all chunks to a single file for embedding
with open("legal_opinion_chunks.json", 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")
