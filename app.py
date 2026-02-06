from pathlib import Path
import os

from fastapi import FastAPI
from pydantic import BaseModel, conlist
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from model.arch import MNISTClassifier

load_dotenv()

CKPT_PATH = os.getenv("MODEL_CHECKPOINT_PATH")
if not CKPT_PATH:
    raise RuntimeError("MODEL_CHECKPOINT_PATH is not set in .env")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEVICE = "cpu"

ckpt_path = Path(CKPT_PATH).expanduser().resolve()
if not ckpt_path.exists():
    raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model = MNISTClassifier(1, 10).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

class PredictRequest(BaseModel):
    pixels: conlist(float, min_length=784, max_length=784)

@app.post("/api/v1/predict")
def predict(req: PredictRequest):
    x = torch.tensor(req.pixels, dtype=torch.float32).view(1, 1, 28, 28)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        digit = int(torch.argmax(probs))

    return {
        "digit": digit,
        "probs": probs.tolist()
    }

@app.get("/health")
def health():
    return {"status": "ok"}