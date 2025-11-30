# llm_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
MODEL_PATH = "checkpoints/llm_gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

app = FastAPI(title="Fine-tuned GPT-2 API")

class Request(BaseModel):
    question: str

@app.post("/generate")
def generate_answer(req: Request):
    # Format the prompt consistently
    prompt = (
        "Question: " + req.question + "\n\n"
        "Answer: Thank you for your question. "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure ending phrase
    if "Let me know if you have any other questions." not in text:
        text += " Let me know if you have any other questions."

    # Return only the answer portion
    answer = text.split("Answer:")[-1].strip()
    return {"answer": answer}
