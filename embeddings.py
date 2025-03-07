#https://www.youtube.com/watch?v=RiaundHy1Xc
from together import Together

client = Together()

response = client.files.upload(file="")
fileId = response.model_dump()["id"]

resp = client.fine_tuning.create(
    model="",
    training_file=fileId,
    n_epoches=3,
    learning_rate=1e-5,
)

print(resp)