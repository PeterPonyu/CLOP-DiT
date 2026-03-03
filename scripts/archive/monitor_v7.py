#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def parse_gpu():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return None, None
        first = out.splitlines()[0]
        parts = [p.strip().replace(" %", "").replace(" MiB", "") for p in first.split(",")]
        util = float(parts[0]) if len(parts) > 0 else None
        mem_mib = float(parts[1]) if len(parts) > 1 else None
        mem_gb = mem_mib / 1024.0 if mem_mib is not None else None
        return util, mem_gb
    except Exception:
        return None, None


def parse_latest_epoch_block(lines):
    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
    train_loss_re = re.compile(r"Train\s+Loss:\s*([0-9.]+)")
    val_proto_re = re.compile(r"Val\s+Proto:\s*([0-9.]+)")

    latest_idx = None
    latest_epoch = None
    total_epochs = None
    for idx, line in enumerate(lines):
        m = epoch_re.search(line)
        if m:
            latest_idx = idx
            latest_epoch = int(m.group(1))
            total_epochs = int(m.group(2))

    if latest_idx is None:
        return None

    block = lines[latest_idx: min(latest_idx + 20, len(lines))]
    train_loss = None
    val_proto = None

    for b in block:
        m = train_loss_re.search(b)
        if m:
            train_loss = float(m.group(1))
        m = val_proto_re.search(b)
        if m:
            val_proto = float(m.group(1))

    if train_loss is None or val_proto is None:
        return None

    return {
        "epoch": latest_epoch,
        "total_epochs": total_epochs,
        "train_loss": train_loss,
        "val_proto_acc": val_proto,
    }


def training_complete(lines):
    tokens = [
        "Training complete",
        "Training finished",
        "All runs complete",
        "Finished CLOP",
    ]
    joined = "\n".join(lines[-300:]) if lines else ""
    return any(tok in joined for tok in tokens)


def print_header():
    print("Epoch | Train Loss | Val Proto Acc | ΔVal    | GPU Util% | VRAM GB")
    print("------|------------|---------------|---------|-----------|--------")


def main():
    parser = argparse.ArgumentParser(description="Non-intrusive CLOP log monitor")
    parser.add_argument("--log", default="logs/v7_train.log", help="Path to training log")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval (seconds)")
    parser.add_argument(
        "--summary", default="logs/v7_monitor_summary.jsonl", help="JSONL summary output"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    last_epoch = None
    prev_val = None
    best_val = float("-inf")
    print_header()

    while True:
        if not log_path.exists():
            print(f"Waiting for log file: {log_path}")
            time.sleep(args.interval)
            continue

        try:
            lines = log_path.read_text(errors="ignore").splitlines()
        except Exception as exc:
            print(f"Log read error: {exc}")
            time.sleep(args.interval)
            continue

        parsed = parse_latest_epoch_block(lines)
        util, vram = parse_gpu()

        if parsed is not None and parsed["epoch"] != last_epoch:
            epoch = parsed["epoch"]
            train_loss = parsed["train_loss"]
            val = parsed["val_proto_acc"]

            delta = None if prev_val is None else (val - prev_val)
            delta_str = "NA" if delta is None else f"{delta:+.4f}"

            row = (
                f"{epoch:>5} | {train_loss:>10.4f} | {val:>13.4f} | {delta_str:>7} | "
                f"{(f'{util:.0f}' if util is not None else 'NA'):>9} | "
                f"{(f'{vram:.2f}' if vram is not None else 'NA'):>6}"
            )

            if delta is not None and delta < -0.002:
                row = f"{YELLOW}{row}{RESET}"

            if val > best_val:
                best_val = val
                row = f"{GREEN}{row}  NEW_BEST{RESET}"

            print(row)
            sys.stdout.flush()

            payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_proto_acc": val,
                "best_so_far": best_val,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

            if epoch % 10 == 0:
                print(
                    f"[summary] epoch={epoch} best_val={best_val:.4f} "
                    f"gpu={util if util is not None else 'NA'} "
                    f"vram_gb={vram if vram is not None else 'NA'}"
                )

            prev_val = val
            last_epoch = epoch

        if training_complete(lines):
            print("Detected 'Training complete' in log. Exiting monitor.")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
