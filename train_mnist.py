import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class Config:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 3
    seed: int = 42
    num_workers: int = 0  # mac: 0 ist ok/stabil
    save_dir: str = "artifacts"
    overfit_one_batch: bool = False  # Debug: sollte schnell Richtung 100% gehen


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        n += x.size(0)

    return total_loss / n, correct / n


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = SmallMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Debug-Modus: wir nehmen genau 1 Batch und trainieren immer wieder darauf
    if cfg.overfit_one_batch:
        x0, y0 = next(iter(train_loader))
        x0, y0 = x0.to(device), y0.to(device)
        print("Overfit-one-batch mode: training on a single batch repeatedly.")

        for epoch in range(cfg.epochs):
            model.train()
            opt.zero_grad()
            logits = model(x0)
            loss = loss_fn(logits, y0)
            loss.backward()
            opt.step()

            acc = (logits.argmax(dim=1) == y0).float().mean().item()
            print(f"epoch {epoch + 1}/{cfg.epochs} | loss {loss.item():.4f} | acc {acc:.4f}")

    else:
        for epoch in range(cfg.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            n = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                running_loss += float(loss) * x.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum())
                n += x.size(0)

            train_loss = running_loss / n
            train_acc = correct / n
            test_loss, test_acc = evaluate(model, test_loader, device)

            print(
                f"epoch {epoch + 1}/{cfg.epochs} | "
                f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f}"
            )

    ckpt_path = save_dir / "mnist_mlp.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--save-dir", type=str, default=Config.save_dir)
    p.add_argument("--overfit-one-batch", action="store_true")
    args = p.parse_args()
    return Config(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        save_dir=args.save_dir,
        overfit_one_batch=args.overfit_one_batch,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
