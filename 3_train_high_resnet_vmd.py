"""
Formal High-frequency ResNet-Attention module for VMD high component forecasting.

定位：
- 承接 highResNet.py 的模型思想：Conv1d 残差块 + 局部注意力 + 时间注意力；
- 改造成正式 train / val / test 实验脚本；
- 输出 high 分支预测结果，供后续与 RF low/mid 结果融合。

推荐运行：
python train_high_resnet_vmd.py --data-dir vmd_data --output-root outputs

期望数据文件（优先使用）：
- vmd_data/train_X_high.npy, vmd_data/val_X_high.npy, vmd_data/test_X_high.npy
- vmd_data/train_y_high.npy, vmd_data/val_y_high.npy, vmd_data/test_y_high.npy

兼容旧命名：
- X_high_train.npy / X_high_val.npy / X_high_test.npy
- y_high_train.npy / y_high_val.npy / y_high_test.npy

输入 shape:
- X_high: (N, window, farms, 1) 或 (N, window, farms)
- y_high: (N, 1, farms) 或 (N, farms)

输出目录：
outputs/high_resnet_vmd/
  checkpoints/best.pt
  metrics/config.json
  metrics/loss_history.json
  metrics/test_metrics.json
  predictions/y_high_pred.npy
  predictions/y_high_true.npy
  attention/high_time_attention.npy
  figures/loss_curve.png
  figures/farm_1_high_prediction.png ...
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. 配置
# =========================================================
@dataclass
class Config:
    data_dir: str = "vmd_data"
    output_root: str = "outputs"
    run_name: str = "high_resnet_vmd"
    seed: int = 42

    batch_size: int = 64
    hidden_dim: int = 32
    num_blocks: int = 3
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30
    patience: int = 8

    max_plot_points: int = 300
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2. 工具函数
# =========================================================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train high-frequency ResNet-Attention branch")
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--num-blocks", type=int, default=Config.num_blocks)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--max-plot-points", type=int, default=Config.max_plot_points)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--device", type=str, default=Config.device)
    args = parser.parse_args()
    return Config(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(run_dir: Path) -> None:
    for sub in ["checkpoints", "metrics", "predictions", "attention", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_first_existing(paths) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def load_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(np.float32)


def normalize_x_shape(x: np.ndarray, path: Path) -> np.ndarray:
    """Return X as (N, T, farms)."""
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim != 3:
        raise ValueError(f"{path} 的 X_high shape 应为 (N,T,farms,1) 或 (N,T,farms)，实际为 {x.shape}")
    return x.astype(np.float32)


def normalize_y_shape(y: np.ndarray, path: Path) -> np.ndarray:
    """Return y as (N, farms)."""
    if y.ndim == 3:
        if y.shape[1] != 1:
            raise ValueError(f"{path} 的 y_high 三维时应为 (N,1,farms)，实际为 {y.shape}")
        y = y[:, 0, :]
    if y.ndim != 2:
        raise ValueError(f"{path} 的 y_high shape 应为 (N,1,farms) 或 (N,farms)，实际为 {y.shape}")
    return y.astype(np.float32)


def resolve_split_files(data_dir: Path, split: str) -> Tuple[Path, Path]:
    """Resolve X_high and y_high file names for a split."""
    x_path = find_first_existing([
        data_dir / f"{split}_X_high.npy",
        data_dir / f"X_high_{split}.npy",
        data_dir / f"{split}_x_high.npy",
    ])
    y_path = find_first_existing([
        data_dir / f"{split}_y_high.npy",
        data_dir / f"y_high_{split}.npy",
        data_dir / f"{split}_Y_high.npy",
    ])

    if x_path is None:
        raise FileNotFoundError(
            f"未找到 {split} 的 X_high 文件。尝试过: "
            f"{data_dir / f'{split}_X_high.npy'}, {data_dir / f'X_high_{split}.npy'}"
        )
    if y_path is None:
        raise FileNotFoundError(
            f"未找到 {split} 的 y_high 文件。高频模块必须用高频目标 y_high，"
            f"不要用完整 y 代替，否则会污染最终融合。尝试过: "
            f"{data_dir / f'{split}_y_high.npy'}, {data_dir / f'y_high_{split}.npy'}"
        )
    return x_path, y_path


def load_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x_path, y_path = resolve_split_files(data_dir, split)
    x = normalize_x_shape(load_array(x_path), x_path)
    y = normalize_y_shape(load_array(y_path), y_path)

    if len(x) != len(y):
        raise ValueError(f"{split} 的 X 和 y 样本数不一致: X={x.shape}, y={y.shape}")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"{split} 的风电场数量不一致: X={x.shape}, y={y.shape}")

    meta = {
        "split": split,
        "x_path": str(x_path),
        "y_path": str(y_path),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
    }
    return x, y, meta


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    out = {
        "overall": {
            "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
            "r2": float(r2_score(y_true_flat, y_pred_flat)),
            "samples": int(y_true.shape[0]),
        },
        "per_farm": {},
    }
    for i in range(y_true.shape[1]):
        t = y_true[:, i]
        p = y_pred[:, i]
        out["per_farm"][f"farm_{i+1}"] = {
            "mae": float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "r2": float(r2_score(t, p)),
            "samples": int(len(t)),
        }
    return out


# =========================================================
# 3. Dataset
# =========================================================
class HighFreqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# =========================================================
# 4. 模型：局部 ResNet + 时间注意力
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.local_attn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        weight = self.local_attn(out)
        out = out * weight
        return self.relu(out + residual)


class HighFreqNet(nn.Module):
    def __init__(self, num_farms: int, hidden_dim: int = 32, num_blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_farms = num_farms
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(num_farms, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])

        # 输出形状为 (B, farms, T)，用于解释每个风电场在高频分支中的时间关注权重。
        self.time_attn = nn.Sequential(
            nn.Conv1d(hidden_dim, num_farms, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_dim, num_farms)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, farms)
        x = self.input_proj(x)        # (B, T, H)
        x = x.permute(0, 2, 1)        # (B, H, T)
        feat = self.blocks(x)         # (B, H, T)

        attn = self.time_attn(feat)   # (B, farms, T)
        time_weight = attn.mean(dim=1, keepdim=True)  # (B, 1, T)
        feat = feat * time_weight
        feat_pool = feat.mean(dim=-1) # (B, H)
        pred = self.head(feat_pool)   # (B, farms)
        return pred, attn


# =========================================================
# 5. 训练与评估
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, _ = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, collect_attention: bool = False):
    model.eval()
    total_loss = 0.0
    total_n = 0
    preds, targets, attentions = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, attn = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
        if collect_attention:
            attentions.append(attn.cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    attn_arr = np.concatenate(attentions, axis=0) if collect_attention and attentions else None
    return total_loss / max(total_n, 1), pred_arr, target_arr, attn_arr


# =========================================================
# 6. 绘图
# =========================================================
def plot_loss_curve(history: Dict, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("High-frequency ResNet Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_prediction_curves(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path, max_points: int) -> None:
    show_points = min(max_points, len(y_true))
    for farm_idx in range(y_true.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:show_points, farm_idx], label="true_high")
        plt.plot(y_pred[:show_points, farm_idx], label="pred_high")
        plt.xlabel("Sample")
        plt.ylabel("Scaled high component")
        plt.title(f"High-frequency Prediction - Farm {farm_idx + 1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"farm_{farm_idx + 1}_high_prediction.png", dpi=220)
        plt.close()


def plot_attention(attn: Optional[np.ndarray], fig_dir: Path) -> None:
    if attn is None or len(attn) == 0:
        return
    sample = attn[0]  # (farms, T)
    plt.figure(figsize=(10, 5))
    for i in range(sample.shape[0]):
        plt.plot(sample[i], label=f"Farm {i + 1}")
    plt.xlabel("Time step")
    plt.ylabel("Attention value")
    plt.title("High-frequency Temporal Attention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "high_time_attention_sample.png", dpi=220)
    plt.close()


# =========================================================
# 7. 主流程
# =========================================================
def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)

    print("读取 high 分支数据...")
    x_train, y_train, train_meta = load_split(data_dir, "train")
    x_val, y_val, val_meta = load_split(data_dir, "val")
    x_test, y_test, test_meta = load_split(data_dir, "test")

    num_farms = x_train.shape[-1]
    window = x_train.shape[1]

    data_meta = {
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
        "num_farms": int(num_farms),
        "window": int(window),
        "note": "high branch uses y_high as target; do not replace with full y.",
    }

    config_dump = asdict(cfg)
    config_dump["num_farms"] = int(num_farms)
    config_dump["window"] = int(window)
    save_json(config_dump, run_dir / "metrics" / "config.json")
    save_json(data_meta, run_dir / "metrics" / "data_meta.json")

    print("X_train:", x_train.shape, "y_train:", y_train.shape)
    print("X_val  :", x_val.shape, "y_val  :", y_val.shape)
    print("X_test :", x_test.shape, "y_test :", y_test.shape)

    train_dataset = HighFreqDataset(x_train, y_train)
    val_dataset = HighFreqDataset(x_val, y_val)
    test_dataset = HighFreqDataset(x_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device(cfg.device)
    model = HighFreqNet(
        num_farms=num_farms,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    start_time = time.time()

    print("\n开始训练 high-frequency ResNet...\n")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.8f} | val_loss={val_loss:.8f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_dump,
                    "best_val_loss": float(best_val),
                    "best_epoch": int(best_epoch),
                },
                run_dir / "checkpoints" / "best.pt",
            )
        else:
            patience_count += 1
            if cfg.patience > 0 and patience_count >= cfg.patience:
                print(f"早停触发：连续 {cfg.patience} 个 epoch 验证集未提升。")
                break

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - start_time
    save_json(history, run_dir / "metrics" / "loss_history.json")
    plot_loss_curve(history, run_dir / "figures" / "loss_curve.png")

    print("\n加载最优模型并评估...")
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_loss, y_val_pred, y_val_true, _ = eval_one_epoch(model, val_loader, criterion, device)
    test_loss, y_test_pred, y_test_true, test_attn = eval_one_epoch(
        model, test_loader, criterion, device, collect_attention=True
    )

    val_metrics = evaluate_metrics(y_val_true, y_val_pred)
    test_metrics = evaluate_metrics(y_test_true, y_test_pred)
    val_metrics["loss_mse"] = float(val_loss)
    test_metrics["loss_mse"] = float(test_loss)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_val_loss"] = float(best_val)
    test_metrics["runtime_seconds"] = float(runtime)
    test_metrics["note"] = "Metrics are computed in scaled high-component domain."

    save_json(val_metrics, run_dir / "metrics" / "val_metrics.json")
    save_json(test_metrics, run_dir / "metrics" / "test_metrics.json")

    np.save(run_dir / "predictions" / "y_high_pred.npy", y_test_pred.astype(np.float32))
    np.save(run_dir / "predictions" / "y_high_true.npy", y_test_true.astype(np.float32))
    np.save(run_dir / "predictions" / "y_high_val_pred.npy", y_val_pred.astype(np.float32))
    np.save(run_dir / "predictions" / "y_high_val_true.npy", y_val_true.astype(np.float32))
    if test_attn is not None:
        np.save(run_dir / "attention" / "high_time_attention.npy", test_attn.astype(np.float32))
        # 也保存平均 attention，便于论文画图。
        np.save(run_dir / "attention" / "high_time_attention_mean.npy", test_attn.mean(axis=0).astype(np.float32))

    # CSV 方便快速查看
    rows = []
    for sample_idx in range(len(y_test_pred)):
        for farm_idx in range(num_farms):
            rows.append({
                "sample_idx": sample_idx,
                "farm_id": farm_idx + 1,
                "y_true_high": float(y_test_true[sample_idx, farm_idx]),
                "y_pred_high": float(y_test_pred[sample_idx, farm_idx]),
                "abs_error": float(abs(y_test_true[sample_idx, farm_idx] - y_test_pred[sample_idx, farm_idx])),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False, encoding="utf-8-sig")

    plot_prediction_curves(y_test_true, y_test_pred, run_dir / "figures", cfg.max_plot_points)
    plot_attention(test_attn, run_dir / "figures")

    overview = {
        "task": "High-frequency VMD branch forecasting with ResNet-Attention",
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_overall": test_metrics["overall"],
        "output_files": {
            "model": str(run_dir / "checkpoints" / "best.pt"),
            "test_metrics": str(run_dir / "metrics" / "test_metrics.json"),
            "y_high_pred": str(run_dir / "predictions" / "y_high_pred.npy"),
            "y_high_true": str(run_dir / "predictions" / "y_high_true.npy"),
            "high_time_attention": str(run_dir / "attention" / "high_time_attention.npy"),
        },
    }
    save_json(overview, run_dir / "metrics" / "overview.json")

    print("\n===== high 分支测试集指标（scaled 高频域）=====")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
