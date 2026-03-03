# CLOP Experiment Registry (v6.3 → v6.4.1)

## Summary

- v6.3 best val_proto_acc: 10.22% (epoch 32)
- v6.4 best val_proto_acc (from log): 5.65%
- v6.4.1 best val_proto_acc: 10.45% (epoch 102)
- v6.4.1 train/val proto gap at best epoch: 3.48x

## Artifact Locations

- Registry JSON: `results/clop_experiment_registry.json`
- Archive root: `models/checkpoints/archive/`
  - `v63/` contains backup checkpoint + history
  - `v64/` contains log-only record
  - `v641/` contains current best checkpoint + history + log

## Version Management Rules (recommended)

1. 每次训练前先创建新 run_id，并指定独立 save_dir（禁止复用 `models/checkpoints`）。
2. 每次训练必须保存 config + history + best_ckpt + log 四件套。
3. 对比实验统一从 registry 读取，不再手工从混合目录找文件。