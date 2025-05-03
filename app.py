from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_query
import os
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    answer = answer_query(req.question)
    return {"answer": answer}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
