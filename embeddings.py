#https://www.youtube.com/watch?v=RiaundHy1Xc
from together import Together
import os

# client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
client = Together(api_key="tgp_v1_4Dgcnv-_rmTV4EIDl48zUpT2FmtcCHV3aGOtm2R9mwU")

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