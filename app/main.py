from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import spacy

from app.bigram_model import BigramModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus, frequency_threshold=1)

try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/gaussian")
def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> List[float]:
    """Sample from a Gaussian distribution with given mean and variance."""
    if variance < 0:
        raise HTTPException(status_code=400, detail="variance must be >= 0")
    std_dev = float(np.sqrt(variance))
    samples = np.random.normal(mean, std_dev, size)
    return samples.tolist()

@app.post("/embed")
def get_embedding(req: EmbeddingRequest):
    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy model not loaded. Install e.g. en_core_web_md")
    token = nlp(req.word.strip())
    if not token or not token[0].has_vector:
        raise HTTPException(status_code=422, detail="No vector for this token with the loaded model.")
    vec = token[0].vector
    return {"word": req.word, "dim": int(vec.shape[0]), "embedding": vec.tolist()}
