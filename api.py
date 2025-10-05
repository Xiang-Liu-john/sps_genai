from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import transforms
from helper_lib.model import get_model
from helper_lib.checkpoints import load_checkpoint

app = FastAPI()

_preproc = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

model = get_model("CNN")
_ckpt_path = "checkpoints/best.pth"
try:
    load_checkpoint(model, optimizer=None, checkpoint_path=_ckpt_path, device='cpu')
except Exception:
    pass
model.eval()

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    img_bytes = file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    x = _preproc(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    return {"class": pred, "label": labels[pred]}

