from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from .dataset import TrajectoryDataset
from .model import SurfaceTracker, TrackerConfig


def train_tracker(
    trajectory_paths: list[str | Path],
    output: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    sequence_length: int = 32,
    device: str | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = TrajectoryDataset(trajectory_paths, sequence_length)
    if len(dataset) < 2:
        raise ValueError("training requires at least two trajectories")
    val_size = max(1, len(dataset) // 10)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    model = SurfaceTracker(TrackerConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    def loss_for(batch):
        observation = batch["observation"].to(device)
        target = batch["delta_local"].to(device)
        mask = batch["mask"].to(device)
        confidence = batch["confidence"].to(device)
        prediction = model(observation)
        variance = prediction["log_variance"].exp()
        displacement = F.smooth_l1_loss(prediction["delta_local"], target, reduction="none") / variance
        displacement = (displacement.mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        confidence_loss = F.binary_cross_entropy_with_logits(
            prediction["confidence_logit"], confidence.clamp(0, 1), reduction="none"
        ).mean()
        return displacement + 0.1 * confidence_loss + 0.005 * prediction["log_variance"].mean()

    metrics = {}
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss = loss_for(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        model.eval()
        with torch.no_grad():
            validation = sum(float(loss_for(batch)) for batch in val_loader) / max(1, len(val_loader))
        metrics = {"epoch": epoch + 1, "train_loss": total / max(1, len(train_loader)), "val_loss": validation}
        print(json.dumps(metrics))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    return metrics
