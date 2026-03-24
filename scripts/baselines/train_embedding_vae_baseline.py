#!/usr/bin/env python3
"""Train a simple VAE baseline on scGPT embeddings and export benchmark artifacts.

This baseline stays in the same embedding space as CLOP-DiT, which makes it a
fair first learned comparator for the multi-method benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingVAE(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _normalise_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def _update_manifest(output_dir: Path, metadata: dict) -> None:
    manifest_path = output_dir.parent / "manifest.json"
    manifest = {"artifact_contract_version": 1, "methods": {}}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest["methods"][metadata["slug"]] = metadata
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EmbeddingVAE baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output-dir", default="results/baselines/embedding_vae")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-per-type", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    emb_path = Path(args.embeddings or cfg.get("embeddings", "data/cached_latents/cell_embeddings_dedup_preprocessed.npy"))
    lbl_path = Path(args.labels or cfg.get("labels", "data/cached_latents/text_group_ids_dedup.npy"))
    epochs = args.epochs or cfg.get("epochs", 60)
    batch_size = args.batch_size or cfg.get("batch_size", 1024)
    hidden_dim = args.hidden_dim or cfg.get("hidden_dim", 512)
    latent_dim = args.latent_dim or cfg.get("latent_dim", 64)
    beta = args.beta if args.beta is not None else cfg.get("beta", 0.01)
    lr = args.lr or cfg.get("lr", 1e-3)
    n_per_type = args.num_per_type or cfg.get("num_per_type", 100)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if not emb_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(f"Missing embeddings or labels: {emb_path} / {lbl_path}")

    embeddings = np.load(emb_path).astype(np.float32)
    labels = np.load(lbl_path).astype(np.int64)
    unique_types = np.sort(np.unique(labels))

    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = EmbeddingVAE(input_dim=embeddings.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_seen = 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            total_recon += float(recon_loss.item()) * bs
            total_kl += float(kl.item()) * bs
            n_seen += bs

        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(n_seen, 1),
                "recon_loss": total_recon / max(n_seen, 1),
                "kl": total_kl / max(n_seen, 1),
            }
        )

    model.eval()
    with torch.no_grad():
        full = torch.from_numpy(embeddings).to(device)
        mu, logvar = model.encode(full)
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()

    gen_embeddings = []
    gen_labels = []
    for type_id in unique_types:
        z_mu = mu_np[labels == type_id]
        z_logvar = logvar_np[labels == type_id]
        if len(z_mu) == 0:
            continue
        mean = z_mu.mean(axis=0)
        std = np.sqrt(np.exp(z_logvar).mean(axis=0) + 1e-8)
        z = np.random.normal(loc=mean, scale=std, size=(n_per_type, latent_dim)).astype(np.float32)
        with torch.no_grad():
            decoded = model.decode(torch.from_numpy(z).to(device)).cpu().numpy()
        decoded = _normalise_rows(decoded)
        gen_embeddings.append(decoded)
        gen_labels.append(np.full(n_per_type, type_id, dtype=np.int64))

    gen_embeddings = np.concatenate(gen_embeddings, axis=0)
    gen_labels = np.concatenate(gen_labels, axis=0)

    np.save(output_dir / "embeddings.npy", gen_embeddings)
    np.save(output_dir / "labels.npy", gen_labels)
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    metadata = {
        "slug": "embedding_vae",
        "display_name": "EmbeddingVAE",
        "family": "learned_baseline",
        "source": "project",
        "seed": seed,
        "input_embeddings": str(emb_path),
        "labels": str(lbl_path),
        "output_embeddings": str(output_dir / "embeddings.npy"),
        "output_labels": str(output_dir / "labels.npy"),
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "beta": beta,
        "lr": lr,
        "num_per_type": n_per_type,
        "artifact_contract_version": 1,
        "fairness_notes": "Uses the same scGPT embedding cache and label set as CLOP-DiT; outputs remain in the same embedding space for direct benchmarking.",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _update_manifest(output_dir, metadata)

    ckpt_dir = Path("models/baselines")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "embedding_vae.pt")


if __name__ == "__main__":
    main()
