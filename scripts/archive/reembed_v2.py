#!/usr/bin/env python3
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer


def load_config_encoder_path(config_path: Path) -> str:
    cfg = yaml.safe_load(config_path.read_text())
    for key in ("sigliP_checkpoint", "siglip_checkpoint", "text_encoder_path"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        f"No text encoder path found in {config_path}. Expected one of: "
        "sigliP_checkpoint, siglip_checkpoint, text_encoder_path"
    )


def load_caption_dict(caption_path: Path) -> OrderedDict:
    raw = json.loads(caption_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Caption JSON must be a dict, got {type(raw)}")

    def key_sort(k: str):
        try:
            return (0, int(k))
        except Exception:
            return (1, str(k))

    ordered = OrderedDict((k, raw[k]) for k in sorted(raw.keys(), key=key_sort))
    return ordered


def encode_captions(texts, model_name_or_path: str, batch_size: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
    model.eval()

    all_embs = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**toks)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy().astype(np.float32))

    elapsed = time.time() - start
    embs = np.concatenate(all_embs, axis=0)
    return embs, elapsed


def main():
    parser = argparse.ArgumentParser(description="Re-embed v2 captions with config-defined text encoder")
    parser.add_argument("--config", default="configs/clop_v641.yaml")
    parser.add_argument("--captions", default="data/cached_latents_v5.2/text_strings_polished.json")
    parser.add_argument("--cache_dir", default="data/cached_latents_v5.2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    captions_path = Path(args.captions)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = load_config_encoder_path(config_path)
    ordered = load_caption_dict(captions_path)

    keys = list(ordered.keys())
    texts = [ordered[k] for k in keys]

    embs, elapsed = encode_captions(texts, encoder_path, args.batch_size, args.device)

    norms = np.linalg.norm(embs, axis=1)
    avg_norm = float(norms.mean())
    if not (0.99 <= avg_norm <= 1.01):
        raise AssertionError(f"avg_norm out of range: {avg_norm:.6f}")

    out_emb = cache_dir / "text_embeddings_v2.npy"
    out_txt = cache_dir / "text_strings_v2.json"
    out_meta = cache_dir / "embed_meta_v2.json"

    np.save(out_emb, embs.astype(np.float32))
    out_txt.write_text(json.dumps(OrderedDict((k, ordered[k]) for k in keys), indent=2))
    out_meta.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": encoder_path,
        "n": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "avg_norm": avg_norm,
    }, indent=2))

    print("N embedded | dim | avg norm | time elapsed")
    print(f"{embs.shape[0]} | {embs.shape[1]} | {avg_norm:.6f} | {elapsed:.2f}s")


if __name__ == "__main__":
    main()
