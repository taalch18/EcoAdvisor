from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .rag import EcoAdvisor

app = FastAPI(title="EcoWise Advisor API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

advisor = None

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def load_model():
    global advisor
    try:
        advisor = EcoAdvisor()
        print("EcoAdvisor Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load EcoAdvisor: {e}")

@app.post("/api/query")
def query_advisor(request: QueryRequest):
    if not advisor:
         raise HTTPException(status_code=503, detail="Model not loaded or failed to initialize")

    result = advisor.ask(request.query)
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": advisor is not None}
