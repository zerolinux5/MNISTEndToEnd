from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary

from tqdm import tqdm
import wandb

from model.arch import MNISTClassifier

# Step through loader and update model, obtain the train loss and correct
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def main():
    config = {
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-3,
        "val_frac": 0.10,
        "seed": 55,
    }

    wandb.init(
        project="mnist-end-to-end",
        config=config
    )
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_ds = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_ds = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    n_total = len(full_ds)
    n_val = int(n_total * config["val_frac"])
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(config["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    # only true for cuda
    should_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=should_pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=should_pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=should_pin_memory,
    )

    model = MNISTClassifier(1, 10).to(device)
    summary_str = str(summary(model, input_size=(config["batch_size"], 1, 28, 28), verbose=2, device=device))
    wandb.log({"model/summary": wandb.Html(f"<pre>{summary_str}</pre>")})
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": dict(config)
                },
                best_path
            )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    if best_path.exists():
        artifact = wandb.Artifact(
            name="mnist-classifier",
            type="model",
            metadata={"test_accuracy": test_acc}
        )
        artifact.add_file(best_path)
        wandb.log_artifact(artifact, aliases=["best"])

    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc})
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()