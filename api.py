"""FastAPI service for the ITRI625 phishing e-mail detector.

Run:  uvicorn api:app --reload        (docs at http://127.0.0.1:8000/docs)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from phishing_core import PhishingClassifier

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = PhishingClassifier.from_dir("artifacts")
    yield
    state.clear()


app = FastAPI(title="Phishing E-mail Detector", version="1.0", lifespan=lifespan)


class EmailIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Raw e-mail text")


class Prediction(BaseModel):
    label: str
    is_phishing: bool
    probability: float
    threshold: float


class WordWeight(BaseModel):
    word: str
    weight: float


class Explanation(BaseModel):
    features: list[WordWeight]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "clf" in state}


@app.post("/predict", response_model=Prediction)
def predict(email: EmailIn):
    return state["clf"].predict(email.text)


@app.post("/explain", response_model=Explanation)
def explain(email: EmailIn):
    # plain `def` -> FastAPI runs this (CPU-heavy) call in a worker thread
    return {"features": state["clf"].explain(email.text, num_features=10, num_samples=500)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000)
