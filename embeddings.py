#https://www.youtube.com/watch?v=RiaundHy1Xc
from together import Together
import os

# client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise RuntimeError("TOGETHER_API_KEY environment variable not set!")
client = Together(api_key=api_key)

# Upload the generated training data
response = client.files.upload(file="training_data.jsonl")
fileId = response.model_dump()["id"]

resp = client.fine_tuning.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    training_file=fileId,
    n_epoches=3,
    learning_rate=1e-5,
)

print(resp)