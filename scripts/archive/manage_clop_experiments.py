#!/usr/bin/env python3
"""Organize CLOP experiment history and generate a diagnostic registry."""

import json
import re
import shutil
from pathlib import Path

ROOT = Path("/home/zeyufu/Desktop/CLOP-DiT")
CKPT = ROOT / "models" / "checkpoints"
LOGS = ROOT / "logs"
ARCHIVE = CKPT / "archive"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

for p in [ARCHIVE, RESULTS, DOCS]:
    p.mkdir(parents=True, exist_ok=True)


def parse_history(path: Path):
    if not path.exists():
        return None
    h = json.loads(path.read_text())
    out = {
        "epochs": len(h.get("train_loss", [])),
        "best_val_proto_acc": None,
        "best_epoch": None,
        "final_val_proto_acc": None,
        "final_val_acc": h.get("val_acc", [None])[-1] if h.get("val_acc") else None,
        "final_val_loss": h.get("val_loss", [None])[-1] if h.get("val_loss") else None,
        "has_train_proto": "train_proto_acc" in h,
    }
    if h.get("val_proto_acc"):
        vals = h["val_proto_acc"]
        out["best_val_proto_acc"] = max(vals)
        out["best_epoch"] = vals.index(max(vals)) + 1
        out["final_val_proto_acc"] = vals[-1]
    if h.get("temperature"):
        out["temp_first"] = h["temperature"][0]
        out["temp_last"] = h["temperature"][-1]
    if h.get("train_proto_acc"):
        tp = h["train_proto_acc"]
        out["best_train_proto_acc"] = max(tp)
        out["final_train_proto_acc"] = tp[-1]
        if out["best_epoch"] and out["best_epoch"] <= len(tp) and out["best_val_proto_acc"]:
            out["gap_train_val_at_best"] = tp[out["best_epoch"] - 1] / out["best_val_proto_acc"]
    return out


def parse_v64_log(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    best_vals = [float(x) for x in re.findall(r"val_proto_acc=([0-9.]+)", txt)]
    epochs = [int(x) for x in re.findall(r"Epoch\s+(\d+)/200", txt)]
    val_proto_lines = [float(x) for x in re.findall(r"Val Proto:\s+([0-9.]+)", txt)]
    train_proto_lines = [float(x) for x in re.findall(r"Train Proto:\s+([0-9.]+)", txt)]

    out = {
        "epochs_seen": max(epochs) if epochs else None,
        "best_val_proto_acc": max(best_vals) if best_vals else None,
        "val_proto_series_tail": val_proto_lines[-10:],
        "train_proto_series_tail": train_proto_lines[-10:],
    }
    if "Early stopping at epoch" in txt:
        m = re.search(r"Early stopping at epoch\s+(\d+)", txt)
        if m:
            out["early_stop_epoch"] = int(m.group(1))
    return out


def safe_copy(src: Path, dst: Path):
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


# Parse experiments
v63 = parse_history(CKPT / "v63_backup" / "clop_history.json")
v641 = parse_history(CKPT / "clop_history.json")
v64 = parse_v64_log(LOGS / "v64_train.log")

registry = {
    "experiments": {
        "v6.3": {
            "history": "models/checkpoints/v63_backup/clop_history.json",
            "best_ckpt": "models/checkpoints/v63_backup/clop_best.pth",
            "log": "logs/clop_training_v6.2.log",
            "metrics": v63,
        },
        "v6.4": {
            "history": None,
            "best_ckpt": None,
            "log": "logs/v64_train.log",
            "metrics": v64,
            "note": "checkpoint/history were overwritten by later runs; recovered from log only",
        },
        "v6.4.1": {
            "history": "models/checkpoints/clop_history.json",
            "best_ckpt": "models/checkpoints/clop_best.pth",
            "log": "logs/v641_train.log",
            "metrics": v641,
        },
    },
    "key_findings": [
        "Validation split has 0 group_id overlap with training split (dataset-level OOD by text groups).",
        "v6.4.1 recovers v6.3 peak val_proto_acc while reducing train/val overfitting gap.",
        "Need stronger experiment artifact retention to avoid overwritten checkpoints/histories.",
    ],
}

# Save registry json
registry_path = RESULTS / "clop_experiment_registry.json"
registry_path.write_text(json.dumps(registry, indent=2))

# Archive snapshots (non-destructive copies)
(ARCHIVE / "v63").mkdir(parents=True, exist_ok=True)
(ARCHIVE / "v64").mkdir(parents=True, exist_ok=True)
(ARCHIVE / "v641").mkdir(parents=True, exist_ok=True)

safe_copy(CKPT / "v63_backup" / "clop_best.pth", ARCHIVE / "v63" / "clop_best.pth")
safe_copy(CKPT / "v63_backup" / "clop_history.json", ARCHIVE / "v63" / "clop_history.json")
safe_copy(LOGS / "v64_train.log", ARCHIVE / "v64" / "v64_train.log")
safe_copy(CKPT / "clop_best.pth", ARCHIVE / "v641" / "clop_best.pth")
safe_copy(CKPT / "clop_history.json", ARCHIVE / "v641" / "clop_history.json")
safe_copy(LOGS / "v641_train.log", ARCHIVE / "v641" / "v641_train.log")

# Markdown report
md = []
md.append("# CLOP Experiment Registry (v6.3 → v6.4.1)\n")
md.append("## Summary\n")
md.append("- v6.3 best val_proto_acc: {:.2f}% (epoch {})".format(100*v63["best_val_proto_acc"], v63["best_epoch"]) if v63 else "- v6.3: missing")
if v64 and v64.get("best_val_proto_acc") is not None:
    md.append("- v6.4 best val_proto_acc (from log): {:.2f}%".format(100*v64["best_val_proto_acc"]))
else:
    md.append("- v6.4: metrics unavailable")
md.append("- v6.4.1 best val_proto_acc: {:.2f}% (epoch {})".format(100*v641["best_val_proto_acc"], v641["best_epoch"]) if v641 else "- v6.4.1: missing")
if v641 and v641.get("gap_train_val_at_best"):
    md.append("- v6.4.1 train/val proto gap at best epoch: {:.2f}x".format(v641["gap_train_val_at_best"]))

md.append("\n## Artifact Locations\n")
md.append("- Registry JSON: `results/clop_experiment_registry.json`")
md.append("- Archive root: `models/checkpoints/archive/`")
md.append("  - `v63/` contains backup checkpoint + history")
md.append("  - `v64/` contains log-only record")
md.append("  - `v641/` contains current best checkpoint + history + log")

md.append("\n## Version Management Rules (recommended)\n")
md.append("1. 每次训练前先创建新 run_id，并指定独立 save_dir（禁止复用 `models/checkpoints`）。")
md.append("2. 每次训练必须保存 config + history + best_ckpt + log 四件套。")
md.append("3. 对比实验统一从 registry 读取，不再手工从混合目录找文件。")

(DOCS / "CLOP_EXPERIMENTS.md").write_text("\n".join(md))

print(f"✓ Wrote {registry_path}")
print(f"✓ Wrote {DOCS / 'CLOP_EXPERIMENTS.md'}")
print(f"✓ Archived artifacts under {ARCHIVE}")
