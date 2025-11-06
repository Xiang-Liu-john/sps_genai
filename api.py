from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import io
import os

from helper_lib_a4.model import EnergyCNN, UNetSmall, DDPM

app = FastAPI(title="CIFAR10 Energy & Diffusion API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Energy Model ----------
energy_model = EnergyCNN().to(device)
energy_ckpt = "checkpoints/energy.pt"
if os.path.exists(energy_ckpt):
    energy_model.load_state_dict(torch.load(energy_ckpt, map_location=device))
    print("Energy model loaded successfully.")
else:
    print(f"⚠️ Energy checkpoint not found at {energy_ckpt}.")
energy_model.eval()

# ---------- Load Diffusion Model ----------
unet = UNetSmall().to(device)
ddpm = DDPM(unet, device=device)
diff_ckpt = "checkpoints/diffusion.pt"
if os.path.exists(diff_ckpt):
    unet.load_state_dict(torch.load(diff_ckpt, map_location=device))
    print("Diffusion model loaded successfully.")
else:
    print(f"Diffusion checkpoint not found at {diff_ckpt}.")
unet.eval()

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

@app.get("/")
def root():
    return {
        "message": "CIFAR10 Energy & Diffusion Models are running!",
        "endpoints": ["/docs", "/energy/score", "/diffusion/sample"]
    }

# ---------- Energy Model Endpoint ----------
@app.post("/energy/score", summary="Compute energy value for uploaded CIFAR-like image")
async def energy_score(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        e = energy_model(x).item()
    return {"energy_score": float(e)}

# ---------- Diffusion Sampling Endpoint ----------
@app.get("/diffusion/sample", summary="Generate CIFAR10-like images (PNG)")
def diffusion_sample(n: int = 16):
    n = max(1, min(int(n), 64))
    with torch.no_grad():
        samples = ddpm.sample(n=n)
        imgs = (samples + 1) / 2  # [0,1]
    buffer = io.BytesIO()
    save_image(imgs, buffer, format="PNG", nrow=min(8, n))
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
