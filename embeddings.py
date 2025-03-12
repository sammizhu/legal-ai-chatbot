#https://www.youtube.com/watch?v=RiaundHy1Xc
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

response = client.files.upload(file="")
fileId = response.model_dump()["id"]

resp = client.fine_tuning.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    training_file=fileId,
    n_epoches=3,
    learning_rate=1e-5,
)

print(resp)