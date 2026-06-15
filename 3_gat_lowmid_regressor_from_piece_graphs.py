"""
Train edge-aware star Graph Attention Network for low/mid prediction.

承接：
1_XunZhaoZuiYouZhi_piece_matching_full_sweep_E1_E6_save_visdata.py  不改
2_trend_piece_matching_from_best_edge.py                         生成 RF wide 特征
2b_build_gat_graph_dataset_from_piece_features.py                生成 GAT 图数据
本脚本作为新的 3 号学习器，用图注意力学习器替代/对比 RF low/mid 回归器。

模型思想：
- 每个目标趋势零件构成一个星型图；
- 目标节点：当前风电场当前趋势零件；
- 邻居节点：Top-k 历史相似趋势零件；
- 边特征：similarity、方向相似度、长度相似度、斜率相似度、time_lag、source_farm、rank 等；
- GAT 注意力权重学习“哪些历史相似片段更值得参考”；
- 输出 low/mid 下一步 delta 或 future_y；
- 保存的 arrays 文件名与 RF 学习器保持一致，可直接供 4_fusion_rf_lowmid_high_resnet.py 使用。

推荐运行：
python 3_gat_lowmid_regressor_from_piece_graphs.py \
  --graph-dir outputs/gat_piece_graph_dataset/graphs \
  --output-root outputs

接入 4 号融合脚本：
python 4_fusion_rf_lowmid_high_resnet.py \
  --lowmid-dir outputs/gat_lowmid_regressor_best_edge \
  --high-dir outputs/high_resnet_vmd \
  --data-dir vmd_data \
  --output-root outputs
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError("本脚本需要 PyTorch。请先安装 torch，或在已安装 PyTorch 的环境中运行。") from exc


@dataclass
class Config:
    graph_dir: str = "outputs/gat_piece_graph_dataset/graphs"
    output_root: str = "outputs"
    run_name: str = "gat_lowmid_regressor_best_edge"

    train_file: str = "train_gat_graphs.npz"
    val_file: str = "val_gat_graphs.npz"
    test_file: str = "test_gat_graphs.npz"

    prediction_target: str = "delta"  # delta / future_y

    hidden_dim: int = 96
    num_heads: int = 4
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 200
    patience: int = 25
    min_delta: float = 1e-6

    loss: str = "huber"  # mse / mae / huber
    huber_delta: float = 1.0

    random_seed: int = 42
    num_workers: int = 0
    device: str = "auto"  # auto / cpu / cuda

    train_with_val: bool = False

    # 输出和可视化
    plot_points: int = 300
    save_attention: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train star-GAT low/mid regressor from trend-piece graph data")
    parser.add_argument("--graph-dir", type=str, default=Config.graph_dir)
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--train-file", type=str, default=Config.train_file)
    parser.add_argument("--val-file", type=str, default=Config.val_file)
    parser.add_argument("--test-file", type=str, default=Config.test_file)
    parser.add_argument("--prediction-target", choices=["delta", "future_y"], default=Config.prediction_target)

    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--num-heads", type=int, default=Config.num_heads)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--min-delta", type=float, default=Config.min_delta)
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default=Config.loss)
    parser.add_argument("--huber-delta", type=float, default=Config.huber_delta)

    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=Config.device)
    parser.add_argument("--train-with-val", action="store_true")
    parser.add_argument("--plot-points", type=int, default=Config.plot_points)
    parser.add_argument("--no-save-attention", action="store_true")
    args = parser.parse_args()

    return Config(
        graph_dir=args.graph_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        prediction_target=args.prediction_target,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        loss=args.loss,
        huber_delta=args.huber_delta,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        device=args.device,
        train_with_val=args.train_with_val,
        plot_points=args.plot_points,
        save_attention=not args.no_save_attention,
    )


def ensure_dirs(run_dir: Path) -> None:
    for sub in ["metrics", "models", "predictions", "arrays", "figures", "tables", "attention"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境没有可用 CUDA。")
        return torch.device("cuda")
    if cfg.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_graph_npz(path: Path, prediction_target: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"未找到图数据文件: {path}")
    data = dict(np.load(path, allow_pickle=False))
    required = ["node_x", "edge_attr", "neighbor_mask", "y_delta", "y_future", "current_y",
                "target_farm", "target_sample_idx", "target_piece_id"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} 缺少必要数组: {missing}")

    if prediction_target == "delta":
        data["y"] = data["y_delta"].astype(np.float32)
    else:
        data["y"] = data["y_future"].astype(np.float32)

    # 保证 dtype
    data["node_x"] = data["node_x"].astype(np.float32)
    data["edge_attr"] = data["edge_attr"].astype(np.float32)
    data["neighbor_mask"] = data["neighbor_mask"].astype(np.float32)
    for k in ["y", "y_delta", "y_future", "current_y"]:
        data[k] = data[k].astype(np.float32)
    return data


class GraphDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data

    def __len__(self) -> int:
        return int(self.data["node_x"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "node_x": torch.from_numpy(self.data["node_x"][idx]),
            "edge_attr": torch.from_numpy(self.data["edge_attr"][idx]),
            "neighbor_mask": torch.from_numpy(self.data["neighbor_mask"][idx]),
            "y": torch.tensor(self.data["y"][idx], dtype=torch.float32),
            "current_y": torch.tensor(self.data["current_y"][idx], dtype=torch.float32),
            "y_future": torch.tensor(self.data["y_future"][idx], dtype=torch.float32),
            "y_delta": torch.tensor(self.data["y_delta"][idx], dtype=torch.float32),
            "target_farm": torch.tensor(self.data["target_farm"][idx], dtype=torch.long),
            "target_sample_idx": torch.tensor(self.data["target_sample_idx"][idx], dtype=torch.long),
            "target_piece_id": torch.tensor(self.data["target_piece_id"][idx], dtype=torch.long),
        }


class EdgeAwareStarGAT(nn.Module):
    """
    不依赖 PyG 的轻量星型 GAT。
    每个样本是一张星型图：
    node_x[:, 0, :] 为 target 节点；
    node_x[:, 1:, :] 为 Top-k source 邻居；
    edge_attr[:, k, :] 为 source_k -> target 的边特征。
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 num_heads={num_heads} 整除。")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.e_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_norm = nn.LayerNorm(hidden_dim * 2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_x: torch.Tensor, edge_attr: torch.Tensor, neighbor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_x: [B, K+1, node_dim]
        # edge_attr: [B, K, edge_dim]
        # neighbor_mask: [B, K]
        h = self.node_encoder(node_x)
        target_h = h[:, 0, :]      # [B, H]
        source_h = h[:, 1:, :]     # [B, K, H]
        edge_h = self.edge_encoder(edge_attr)

        bsz, top_k, _ = source_h.shape
        q = self.q_proj(target_h).view(bsz, self.num_heads, self.head_dim)             # [B, heads, D]
        k = self.k_proj(source_h).view(bsz, top_k, self.num_heads, self.head_dim)      # [B, K, heads, D]
        v = self.v_proj(source_h).view(bsz, top_k, self.num_heads, self.head_dim)
        e = self.e_proj(edge_h).view(bsz, top_k, self.num_heads, self.head_dim)

        # 边特征注入 key，体现 similarity / lag / source_farm 对注意力分数的影响。
        score = (q[:, None, :, :] * (k + e)).sum(dim=-1) / math.sqrt(self.head_dim)    # [B, K, heads]
        score = torch.nn.functional.leaky_relu(score, negative_slope=0.2)

        mask = neighbor_mask[:, :, None] > 0.5
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)                                            # [B, K, heads]

        context = (alpha[:, :, :, None] * v).sum(dim=1).reshape(bsz, self.hidden_dim)  # [B, H]
        fused = self.out_norm(torch.cat([target_h, context], dim=-1))
        pred = self.regressor(fused).squeeze(-1)
        return pred, alpha


def build_loss(cfg: Config):
    if cfg.loss == "mse":
        return nn.MSELoss()
    if cfg.loss == "mae":
        return nn.L1Loss()
    return nn.HuberLoss(delta=cfg.huber_delta)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "r2": None, "samples": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else None,
        "samples": int(len(yt)),
    }


def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> float:
    model.train(train)
    losses = []
    for batch in loader:
        node_x = batch["node_x"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["neighbor_mask"].to(device)
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
        pred, _ = model(node_x, edge_attr, mask)
        loss = criterion(pred, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()) * len(y))
    return float(np.sum(losses) / max(len(loader.dataset), 1))


@torch.no_grad()
def predict(model, loader, device, prediction_target: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        node_x = batch["node_x"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["neighbor_mask"].to(device)
        pred_target, alpha = model(node_x, edge_attr, mask)

        pred_target_np = pred_target.detach().cpu().numpy()
        current_np = batch["current_y"].numpy()
        y_future_np = batch["y_future"].numpy()
        y_delta_np = batch["y_delta"].numpy()

        if prediction_target == "delta":
            pred_delta = pred_target_np
            pred_future = current_np + pred_delta
        else:
            pred_future = pred_target_np
            pred_delta = pred_future - current_np

        alpha_np = alpha.detach().cpu().numpy()  # [B,K,heads]
        for i in range(len(pred_target_np)):
            row = {
                "target_farm": int(batch["target_farm"][i].item()),
                "target_sample_idx": int(batch["target_sample_idx"][i].item()),
                "target_piece_id": int(batch["target_piece_id"][i].item()),
                "current_y_lowmid": float(current_np[i]),
                "target_future_y_lowmid": float(y_future_np[i]),
                "target_future_delta_lowmid": float(y_delta_np[i]),
                "gat_pred_target": float(pred_target_np[i]),
                "gat_pred_future_y_lowmid": float(pred_future[i]),
                "gat_pred_delta_lowmid": float(pred_delta[i]),
                "valid_neighbor_count": float(batch["neighbor_mask"][i].sum().item()),
            }
            # 保存每个 rank 的平均注意力和各 head 注意力，便于图注意力可解释性分析。
            mean_alpha = alpha_np[i].mean(axis=1)
            for k_idx, val in enumerate(mean_alpha, start=1):
                row[f"attn_rank_{k_idx}_mean"] = float(val)
            for k_idx in range(alpha_np.shape[1]):
                for h_idx in range(alpha_np.shape[2]):
                    row[f"attn_rank_{k_idx + 1}_head_{h_idx + 1}"] = float(alpha_np[i, k_idx, h_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_sample_predictions(row_pred: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "current_y_lowmid": "mean",
        "target_future_y_lowmid": "mean",
        "target_future_delta_lowmid": "mean",
        "gat_pred_future_y_lowmid": "mean",
        "gat_pred_delta_lowmid": "mean",
        "valid_neighbor_count": "mean",
    }
    for col in row_pred.columns:
        if col.startswith("attn_rank_") and col.endswith("_mean"):
            agg_dict[col] = "mean"

    out = (
        row_pred
        .groupby(["target_farm", "target_sample_idx"], as_index=False)
        .agg(agg_dict)
        .sort_values(["target_farm", "target_sample_idx"])
        .reset_index(drop=True)
    )
    out["gat_abs_error_lowmid"] = np.abs(out["target_future_y_lowmid"] - out["gat_pred_future_y_lowmid"])
    out["gat_squared_error_lowmid"] = (out["target_future_y_lowmid"] - out["gat_pred_future_y_lowmid"]) ** 2
    return out


def metrics_for_prediction(row_pred: pd.DataFrame, sample_pred: pd.DataFrame) -> Dict:
    out = {
        "row_level_future_y": evaluate_regression(row_pred["target_future_y_lowmid"], row_pred["gat_pred_future_y_lowmid"]),
        "row_level_delta": evaluate_regression(row_pred["target_future_delta_lowmid"], row_pred["gat_pred_delta_lowmid"]),
        "sample_level_future_y": evaluate_regression(sample_pred["target_future_y_lowmid"], sample_pred["gat_pred_future_y_lowmid"]),
        "sample_level_delta": evaluate_regression(sample_pred["target_future_delta_lowmid"], sample_pred["gat_pred_delta_lowmid"]),
        "per_farm_sample_level": {},
    }
    for farm_id, sub in sample_pred.groupby("target_farm"):
        out["per_farm_sample_level"][f"farm_{int(farm_id)}"] = evaluate_regression(
            sub["target_future_y_lowmid"], sub["gat_pred_future_y_lowmid"]
        )
    return out


def save_prediction_arrays(sample_pred: pd.DataFrame, split_name: str, run_dir: Path) -> Dict[str, str]:
    if sample_pred.empty:
        return {}
    max_sample = int(sample_pred["target_sample_idx"].max())
    max_farm = int(sample_pred["target_farm"].max())
    y_true = np.full((max_sample + 1, max_farm), np.nan, dtype=np.float32)
    y_pred = np.full((max_sample + 1, max_farm), np.nan, dtype=np.float32)
    y_current = np.full((max_sample + 1, max_farm), np.nan, dtype=np.float32)
    y_delta_pred = np.full((max_sample + 1, max_farm), np.nan, dtype=np.float32)

    for _, row in sample_pred.iterrows():
        i = int(row["target_sample_idx"])
        f = int(row["target_farm"]) - 1
        y_true[i, f] = float(row["target_future_y_lowmid"])
        y_pred[i, f] = float(row["gat_pred_future_y_lowmid"])
        y_current[i, f] = float(row["current_y_lowmid"])
        y_delta_pred[i, f] = float(row["gat_pred_delta_lowmid"])

    paths = {
        "y_true_lowmid_scaled": run_dir / "arrays" / f"{split_name}_y_true_lowmid_scaled.npy",
        "y_pred_lowmid_scaled": run_dir / "arrays" / f"{split_name}_y_pred_lowmid_scaled.npy",
        "y_current_lowmid_scaled": run_dir / "arrays" / f"{split_name}_y_current_lowmid_scaled.npy",
        "y_pred_delta_lowmid_scaled": run_dir / "arrays" / f"{split_name}_y_pred_delta_lowmid_scaled.npy",
    }
    np.save(paths["y_true_lowmid_scaled"], y_true)
    np.save(paths["y_pred_lowmid_scaled"], y_pred)
    np.save(paths["y_current_lowmid_scaled"], y_current)
    np.save(paths["y_pred_delta_lowmid_scaled"], y_delta_pred)
    return {k: str(v) for k, v in paths.items()}


def plot_prediction_curves(sample_pred: pd.DataFrame, split_name: str, run_dir: Path, plot_points: int) -> None:
    if sample_pred.empty:
        return
    for farm_id, sub in sample_pred.groupby("target_farm"):
        sub = sub.sort_values("target_sample_idx")
        n = min(plot_points, len(sub))
        plt.figure(figsize=(11, 4.5))
        plt.plot(sub["target_sample_idx"].iloc[:n], sub["target_future_y_lowmid"].iloc[:n], label="true_lowmid")
        plt.plot(sub["target_sample_idx"].iloc[:n], sub["gat_pred_future_y_lowmid"].iloc[:n], label="gat_pred_lowmid")
        plt.xlabel("Sample")
        plt.ylabel("Scaled low/mid")
        plt.title(f"{split_name} Star-GAT Low/Mid Prediction - Farm {int(farm_id)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / f"{split_name}_farm_{int(farm_id)}_gat_prediction.png", dpi=220)
        plt.close()


def plot_training_history(history: List[Dict], run_dir: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    df.to_csv(run_dir / "tables" / "training_history.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAT Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "training_history.png", dpi=220)
    plt.close()


def save_attention_summary(sample_pred: pd.DataFrame, split_name: str, run_dir: Path) -> None:
    attn_cols = [c for c in sample_pred.columns if c.startswith("attn_rank_") and c.endswith("_mean")]
    if not attn_cols:
        return
    summary = []
    for col in attn_cols:
        rank = int(col.split("_")[2])
        summary.append({
            "split": split_name,
            "rank": rank,
            "mean_attention": float(sample_pred[col].mean()),
            "std_attention": float(sample_pred[col].std()),
        })
    df = pd.DataFrame(summary).sort_values("rank")
    df.to_csv(run_dir / "attention" / f"{split_name}_attention_by_rank.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 4))
    plt.bar(df["rank"], df["mean_attention"])
    plt.xlabel("Top-k rank")
    plt.ylabel("Mean attention weight")
    plt.title(f"{split_name} GAT Attention by Top-k Rank")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / f"{split_name}_attention_by_rank.png", dpi=220)
    plt.close()


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)
    device = pick_device(cfg)

    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)
    save_json(asdict(cfg), run_dir / "metrics" / "config.json")

    graph_dir = Path(cfg.graph_dir)
    train_data = load_graph_npz(graph_dir / cfg.train_file, cfg.prediction_target)
    val_data = load_graph_npz(graph_dir / cfg.val_file, cfg.prediction_target)
    test_data = load_graph_npz(graph_dir / cfg.test_file, cfg.prediction_target)

    if cfg.train_with_val:
        fit_data = {}
        for key in train_data.keys():
            if isinstance(train_data[key], np.ndarray) and key in val_data and train_data[key].ndim >= 1:
                fit_data[key] = np.concatenate([train_data[key], val_data[key]], axis=0)
        train_note = "train+val"
    else:
        fit_data = train_data
        train_note = "train_only"

    train_loader = DataLoader(GraphDataset(fit_data), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(GraphDataset(val_data), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    node_dim = int(train_data["node_x"].shape[-1])
    edge_dim = int(train_data["edge_attr"].shape[-1])

    model = EdgeAwareStarGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)

    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("===== Train Star-GAT Low/Mid Regressor =====")
    print(f"device={device}, train_mode={train_note}")
    print(f"node_dim={node_dim}, edge_dim={edge_dim}, train_graphs={len(GraphDataset(fit_data))}, val_graphs={len(GraphDataset(val_data))}")

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict] = []
    best_path = run_dir / "models" / "best_gat_lowmid.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val - cfg.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
            }, best_path)
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f}@{best_epoch}")

        if bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    plot_training_history(history, run_dir)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    split_loaders = {
        "train": DataLoader(GraphDataset(train_data), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "val": val_loader,
        "test": test_loader,
    }

    all_metrics = {}
    output_arrays = {}

    for split_name, loader in split_loaders.items():
        row_pred = predict(model, loader, device, cfg.prediction_target)
        sample_pred = aggregate_sample_predictions(row_pred)

        row_pred.to_csv(run_dir / "predictions" / f"{split_name}_row_predictions.csv", index=False, encoding="utf-8-sig")
        sample_pred.to_csv(run_dir / "predictions" / f"{split_name}_sample_predictions.csv", index=False, encoding="utf-8-sig")

        metrics = metrics_for_prediction(row_pred, sample_pred)
        all_metrics[split_name] = metrics
        save_json(metrics, run_dir / "metrics" / f"{split_name}_metrics.json")
        output_arrays[split_name] = save_prediction_arrays(sample_pred, split_name, run_dir)

        plot_prediction_curves(sample_pred, split_name, run_dir, cfg.plot_points)
        if cfg.save_attention:
            save_attention_summary(sample_pred, split_name, run_dir)

        m = metrics["sample_level_future_y"]
        print(f"{split_name}: MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, R2={m['r2']}")

    overview = {
        "task": "Star-GAT low/mid regression from trend-piece Top-k graph",
        "run_dir": str(run_dir),
        "graph_dir": str(graph_dir),
        "train_mode": train_note,
        "device": str(device),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_heads": cfg.num_heads,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "main_val_metric_sample_level_future_y": all_metrics["val"]["sample_level_future_y"],
        "main_test_metric_sample_level_future_y": all_metrics["test"]["sample_level_future_y"],
        "output_files": {
            "model": str(best_path),
            "test_row_predictions": str(run_dir / "predictions" / "test_row_predictions.csv"),
            "test_sample_predictions": str(run_dir / "predictions" / "test_sample_predictions.csv"),
            "test_y_pred_lowmid_scaled": output_arrays.get("test", {}).get("y_pred_lowmid_scaled"),
            "test_y_true_lowmid_scaled": output_arrays.get("test", {}).get("y_true_lowmid_scaled"),
            "test_attention_by_rank": str(run_dir / "attention" / "test_attention_by_rank.csv"),
        },
    }
    save_json(all_metrics, run_dir / "metrics" / "all_metrics.json")
    save_json(overview, run_dir / "metrics" / "overview.json")

    print("\n===== 完成 =====")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
