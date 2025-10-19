# api.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
from torchvision.utils import save_image
import io
import os
from helper_lib_new.model import GeneratorMNIST

app = FastAPI(title="MNIST GAN API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = GeneratorMNIST().to(device)

ckpt_path = "./checkpoints/gan_mnist.pth"
try:
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["G"])
        print("✅ Generator load success")
    else:
        print(f"⚠️ Checkpoint not found at {ckpt_path}")
except Exception as e:
    print("⚠️ Cannot load model:", e)

generator.eval()

@app.get("/")
def root():
    return {"message": "MNIST GAN API is running", "endpoints": ["/docs", "/generate?n=16"]}

@app.get("/generate", summary="Generate MNIST-like digits image grid (PNG)")
def generate_digit(n: int = 16):
    n = max(1, min(int(n), 64)) 
    z = torch.randn(n, 100, device=device)
    with torch.no_grad():
        fake = generator(z)           # [-1,1]
        imgs = (fake + 1) / 2         # [0,1]

    buffer = io.BytesIO()

    save_image(imgs, buffer, format="PNG", nrow=int(min(8, n)))
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
