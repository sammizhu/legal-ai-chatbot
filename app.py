from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import answer_query

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    answer = answer_query(req.question)
    return {"answer": answer}
