from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrackerConfig:
    observation_size: int = 13
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.1


class SurfaceTracker(nn.Module):
    """GRU belief-state baseline with displacement, confidence, and uncertainty heads."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.observation_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.core = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0,
        )
        self.delta_head = nn.Linear(config.hidden_size, 3)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        self.log_variance_head = nn.Linear(config.hidden_size, 3)

    def forward(self, observations: torch.Tensor, state: torch.Tensor | None = None):
        encoded = self.encoder(observations)
        belief, state = self.core(encoded, state)
        return {
            "delta_local": self.delta_head(belief),
            "confidence_logit": self.confidence_head(belief).squeeze(-1),
            "log_variance": self.log_variance_head(belief).clamp(-6, 4),
            "state": state,
        }

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "SurfaceTracker":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(TrackerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


def local_frame(tangent: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    e1 = nn.functional.normalize(tangent, dim=-1)
    e3 = nn.functional.normalize(normal, dim=-1)
    e2 = nn.functional.normalize(torch.linalg.cross(e3, e1, dim=-1), dim=-1)
    e1 = nn.functional.normalize(torch.linalg.cross(e2, e3, dim=-1), dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)
