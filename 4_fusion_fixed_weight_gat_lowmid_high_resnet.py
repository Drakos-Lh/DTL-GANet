"""
Fixed-weight fusion module: GAT low/mid + High-frequency ResNet

作用：
1. 读取 low/mid 分支的 GAT 预测结果；
2. 读取 high 分支的 ResNet 预测结果；
3. 在验证集上搜索一组固定融合系数；
4. 在测试集上使用该固定系数融合：
      y_final_scaled = w_lowmid * y_lowmid_pred + w_high * y_high_pred
   其中 w_lowmid 与 w_high 是全测试集共享的固定缩放系数；
5. 同时保存直接相加融合：
      y_add_scaled = y_lowmid_pred + y_high_pred
   作为基础对照；
6. 优先使用 vmd_data/{split}_y_full.npy 作为最终真实值；如果不存在，则使用 y_lowmid_true + y_high_true；
7. 反标准化回原始功率尺度；
8. 保存指标、数组、CSV 明细和预测曲线图。

推荐运行：
python 4_fusion_fixed_weight_gat_lowmid_high_resnet.py \
  --lowmid-dir outputs/gat_lowmid_regressor_best_edge \
  --high-dir outputs/high_resnet_vmd \
  --data-dir vmd_data \
  --output-root outputs \
  --weight-search-split val \
  --eval-split test

默认输出目录：
outputs/fusion_fixed_weight_gat_lowmid_high_resnet

说明：
- 本脚本中的“固定加权”不是 softmax 凸组合。low/mid 与 high 是 VMD 分量，
  更合理的固定融合方式是学习/搜索两个分量的固定缩放系数，而不是在两个分量之间二选一。
- 若希望只修正高频分支，可设置 --search-mode high_only，此时 w_lowmid 固定为 1，
  只搜索 y = y_lowmid + w_high * y_high。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. 配置
# =========================================================
@dataclass
class Config:
    lowmid_dir: str = "outputs/gat_lowmid_regressor_best_edge"
    high_dir: str = "outputs/high_resnet_vmd"
    data_dir: str = "vmd_data"
    output_root: str = "outputs"
    run_name: str = "fusion_fixed_weight_gat_lowmid_high_resnet"

    weight_search_split: str = "val"
    eval_split: str = "test"
    num_farms: int = 6
    max_plot_points: int = 300

    # 当 low/mid 与 high 的样本长度不一致时的处理方式：
    # error: 直接报错，最严谨；trim_head: 保留前 min_len；trim_tail: 保留后 min_len。
    align_mode: str = "error"
    prefer_full_true: bool = True

    # 固定权重搜索配置。
    # two_dim: 同时搜索 w_lowmid 与 w_high；
    # high_only: w_lowmid 固定为 1，只搜索 w_high，适合作为“高频残差修正系数”基线。
    search_mode: str = "two_dim"
    metric: str = "rmse"  # rmse 或 mae
    weight_min: float = 0.5
    weight_max: float = 1.5
    weight_step: float = 0.01


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fixed-weight fuse GAT low/mid prediction and High ResNet prediction")
    parser.add_argument("--lowmid-dir", type=str, default=Config.lowmid_dir)
    parser.add_argument("--high-dir", type=str, default=Config.high_dir)
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--weight-search-split", type=str, default=Config.weight_search_split, choices=["val", "test"])
    parser.add_argument("--eval-split", type=str, default=Config.eval_split, choices=["val", "test"])
    parser.add_argument("--num-farms", type=int, default=Config.num_farms)
    parser.add_argument("--max-plot-points", type=int, default=Config.max_plot_points)
    parser.add_argument("--align-mode", type=str, default=Config.align_mode, choices=["error", "trim_head", "trim_tail"])
    parser.add_argument("--no-prefer-full-true", action="store_true", help="不优先使用 vmd_data/{split}_y_full.npy")

    parser.add_argument("--search-mode", type=str, default=Config.search_mode, choices=["two_dim", "high_only"])
    parser.add_argument("--metric", type=str, default=Config.metric, choices=["rmse", "mae"])
    parser.add_argument("--weight-min", type=float, default=Config.weight_min)
    parser.add_argument("--weight-max", type=float, default=Config.weight_max)
    parser.add_argument("--weight-step", type=float, default=Config.weight_step)

    args = parser.parse_args()
    return Config(
        lowmid_dir=args.lowmid_dir,
        high_dir=args.high_dir,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        weight_search_split=args.weight_search_split,
        eval_split=args.eval_split,
        num_farms=args.num_farms,
        max_plot_points=args.max_plot_points,
        align_mode=args.align_mode,
        prefer_full_true=not args.no_prefer_full_true,
        search_mode=args.search_mode,
        metric=args.metric,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        weight_step=args.weight_step,
    )


# =========================================================
# 2. 基础工具
# =========================================================
def ensure_dirs(run_dir: Path) -> None:
    for sub in ["metrics", "arrays", "predictions", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_npy(path: Path, required: bool = True) -> Optional[np.ndarray]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"未找到文件: {path}")
        return None
    return np.load(path)


def as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """统一转成 (N, farms)。"""
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(f"{name} 应为 (N, farms) 或 (N, 1, farms)，实际 shape={arr.shape}")
    return arr.astype(np.float32)


def resolve_lowmid_array_dir(lowmid_dir: Path) -> Path:
    if (lowmid_dir / "arrays").exists():
        return lowmid_dir / "arrays"
    return lowmid_dir


def resolve_high_prediction_dir(high_dir: Path) -> Path:
    if (high_dir / "predictions").exists():
        return high_dir / "predictions"
    return high_dir


def resolve_lowmid_paths(lowmid_dir: Path, split: str) -> Tuple[Path, Path]:
    arr_dir = resolve_lowmid_array_dir(lowmid_dir)
    pred_candidates = [
        arr_dir / f"{split}_y_pred_lowmid_scaled.npy",
        arr_dir / f"{split}_y_lowmid_pred_scaled.npy",
        arr_dir / f"{split}_y_pred.npy",
    ]
    true_candidates = [
        arr_dir / f"{split}_y_true_lowmid_scaled.npy",
        arr_dir / f"{split}_y_lowmid_true_scaled.npy",
        arr_dir / f"{split}_y_true.npy",
    ]
    pred_path = next((p for p in pred_candidates if p.exists()), pred_candidates[0])
    true_path = next((p for p in true_candidates if p.exists()), true_candidates[0])
    return pred_path, true_path


def resolve_high_paths(high_dir: Path, split: str) -> Tuple[Path, Path]:
    pred_dir = resolve_high_prediction_dir(high_dir)
    if split == "test":
        pred_candidates = [
            pred_dir / "y_high_pred.npy",
            pred_dir / "test_y_high_pred.npy",
            pred_dir / "y_pred_high.npy",
        ]
        true_candidates = [
            pred_dir / "y_high_true.npy",
            pred_dir / "test_y_high_true.npy",
            pred_dir / "y_true_high.npy",
        ]
    else:
        pred_candidates = [
            pred_dir / "y_high_val_pred.npy",
            pred_dir / "val_y_high_pred.npy",
            pred_dir / "y_val_high_pred.npy",
            pred_dir / "val_pred.npy",
        ]
        true_candidates = [
            pred_dir / "y_high_val_true.npy",
            pred_dir / "val_y_high_true.npy",
            pred_dir / "y_val_high_true.npy",
            pred_dir / "val_true.npy",
        ]
    pred_path = next((p for p in pred_candidates if p.exists()), pred_candidates[0])
    true_path = next((p for p in true_candidates if p.exists()), true_candidates[0])
    return pred_path, true_path


def load_scaler(data_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    candidates = [
        (data_dir / "scaler_mean.npy", data_dir / "scaler_scale.npy", "scaler_mean/scaler_scale"),
        (data_dir / "data_mean.npy", data_dir / "data_std.npy", "data_mean/data_std"),
        (Path("data_mean.npy"), Path("data_std.npy"), "root:data_mean/data_std"),
    ]
    for mean_path, scale_path, source in candidates:
        if mean_path.exists() and scale_path.exists():
            mean = np.load(mean_path).reshape(-1).astype(np.float32)
            scale = np.load(scale_path).reshape(-1).astype(np.float32)
            return mean, scale, source
    return None, None, "not_found"


def inverse_scale(arr: np.ndarray, mean: Optional[np.ndarray], scale: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mean is None or scale is None:
        return None
    if arr.shape[1] != len(mean) or arr.shape[1] != len(scale):
        raise ValueError(f"反标准化参数维度不匹配: arr={arr.shape}, mean={mean.shape}, scale={scale.shape}")
    return arr * scale.reshape(1, -1) + mean.reshape(1, -1)


def load_full_true(data_dir: Path, split: str) -> Optional[np.ndarray]:
    candidates = [
        data_dir / f"{split}_y_full.npy",
        data_dir / f"y_{split}_full.npy",
        data_dir / f"{split}_y.npy",
    ]
    for path in candidates:
        if path.exists():
            return as_2d(np.load(path), f"full_true:{path}")
    return None


def align_arrays(arrays: Dict[str, np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    lengths = {name: arr.shape[0] for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) == 1:
        return arrays

    if mode == "error":
        raise ValueError(
            "样本长度不一致，无法直接融合。"
            f"各数组长度: {lengths}。"
            "请检查 low/mid 与 high 是否来自同一套滑窗，或使用 --align-mode trim_head/trim_tail。"
        )

    min_len = min(unique_lengths)
    out = {}
    for name, arr in arrays.items():
        if mode == "trim_head":
            out[name] = arr[:min_len]
        elif mode == "trim_tail":
            out[name] = arr[-min_len:]
        else:
            raise ValueError(f"未知 align_mode: {mode}")
    return out


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"指标计算 shape 不一致: y_true={y_true.shape}, y_pred={y_pred.shape}")

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    out = {
        "overall": {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2": float(r2_score(yt, yp)),
            "samples": int(y_true.shape[0] * y_true.shape[1]),
        },
        "per_farm": {},
    }

    for i in range(y_true.shape[1]):
        t = y_true[:, i]
        p = y_pred[:, i]
        out["per_farm"][f"farm_{i + 1}"] = {
            "mae": float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "r2": float(r2_score(t, p)),
            "samples": int(len(t)),
        }
    return out


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    if metric == "mae":
        return float(mean_absolute_error(yt, yp))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(yt, yp)))
    raise ValueError(f"未知 metric: {metric}")


# =========================================================
# 3. 数据加载与固定权重搜索
# =========================================================
def load_branch_data(cfg: Config, split: str) -> Dict[str, np.ndarray | str]:
    lowmid_dir = Path(cfg.lowmid_dir)
    high_dir = Path(cfg.high_dir)
    data_dir = Path(cfg.data_dir)

    lowmid_pred_path, lowmid_true_path = resolve_lowmid_paths(lowmid_dir, split)
    high_pred_path, high_true_path = resolve_high_paths(high_dir, split)

    print(f"\n===== Load split={split} =====")
    print(f"lowmid_pred: {lowmid_pred_path}")
    print(f"lowmid_true: {lowmid_true_path}")
    print(f"high_pred  : {high_pred_path}")
    print(f"high_true  : {high_true_path}")

    y_lowmid_pred = as_2d(load_npy(lowmid_pred_path), "y_lowmid_pred")
    y_lowmid_true = as_2d(load_npy(lowmid_true_path), "y_lowmid_true")
    y_high_pred = as_2d(load_npy(high_pred_path), "y_high_pred")
    y_high_true = as_2d(load_npy(high_true_path), "y_high_true")

    arrays = align_arrays(
        {
            "y_lowmid_pred": y_lowmid_pred,
            "y_lowmid_true": y_lowmid_true,
            "y_high_pred": y_high_pred,
            "y_high_true": y_high_true,
        },
        cfg.align_mode,
    )
    y_lowmid_pred = arrays["y_lowmid_pred"]
    y_lowmid_true = arrays["y_lowmid_true"]
    y_high_pred = arrays["y_high_pred"]
    y_high_true = arrays["y_high_true"]

    if y_lowmid_pred.shape[1] != cfg.num_farms or y_high_pred.shape[1] != cfg.num_farms:
        raise ValueError(
            f"风电场数量不一致: lowmid={y_lowmid_pred.shape}, high={y_high_pred.shape}, num_farms={cfg.num_farms}"
        )

    y_add_pred_scaled = y_lowmid_pred + y_high_pred

    full_true_scaled = load_full_true(data_dir, split) if cfg.prefer_full_true else None
    true_source = "lowmid_true_plus_high_true"
    if full_true_scaled is not None:
        full_true_scaled = as_2d(full_true_scaled, "full_true_scaled")
        aligned = align_arrays({"full_true_scaled": full_true_scaled, "pred": y_add_pred_scaled}, cfg.align_mode)
        full_true_scaled = aligned["full_true_scaled"]
        pred_aligned = aligned["pred"]
        if len(pred_aligned) != len(y_add_pred_scaled):
            arrays = align_arrays(
                {
                    "y_lowmid_pred": y_lowmid_pred,
                    "y_lowmid_true": y_lowmid_true,
                    "y_high_pred": y_high_pred,
                    "y_high_true": y_high_true,
                    "y_add_pred_scaled": y_add_pred_scaled,
                    "full_true_scaled": full_true_scaled,
                },
                cfg.align_mode,
            )
            y_lowmid_pred = arrays["y_lowmid_pred"]
            y_lowmid_true = arrays["y_lowmid_true"]
            y_high_pred = arrays["y_high_pred"]
            y_high_true = arrays["y_high_true"]
            y_add_pred_scaled = arrays["y_add_pred_scaled"]
            full_true_scaled = arrays["full_true_scaled"]
        y_final_true_scaled = full_true_scaled
        true_source = f"{cfg.data_dir}/{split}_y_full.npy"
    else:
        y_final_true_scaled = y_lowmid_true + y_high_true

    return {
        "y_lowmid_pred": y_lowmid_pred,
        "y_lowmid_true": y_lowmid_true,
        "y_high_pred": y_high_pred,
        "y_high_true": y_high_true,
        "y_add_pred_scaled": y_add_pred_scaled,
        "y_final_true_scaled": y_final_true_scaled,
        "true_source": true_source,
        "lowmid_pred_path": str(lowmid_pred_path),
        "lowmid_true_path": str(lowmid_true_path),
        "high_pred_path": str(high_pred_path),
        "high_true_path": str(high_true_path),
    }


def make_weight_grid(weight_min: float, weight_max: float, weight_step: float) -> np.ndarray:
    if weight_step <= 0:
        raise ValueError("weight_step 必须大于 0")
    if weight_max < weight_min:
        raise ValueError("weight_max 必须不小于 weight_min")
    n = int(round((weight_max - weight_min) / weight_step))
    grid = weight_min + np.arange(n + 1, dtype=np.float32) * weight_step
    # 避免浮点误差导致最后一个点略超出范围。
    grid = grid[grid <= weight_max + 1e-8]
    return np.round(grid, 8)


def search_fixed_weights(
    y_true: np.ndarray,
    y_lowmid_pred: np.ndarray,
    y_high_pred: np.ndarray,
    cfg: Config,
) -> Tuple[float, float, pd.DataFrame]:
    """在指定 split 上搜索固定融合权重。"""
    weights = make_weight_grid(cfg.weight_min, cfg.weight_max, cfg.weight_step)
    records = []
    best_score = float("inf")
    best_w_lowmid = 1.0
    best_w_high = 1.0

    if cfg.search_mode == "high_only":
        for w_high in weights:
            pred = y_lowmid_pred + float(w_high) * y_high_pred
            score = metric_value(y_true, pred, cfg.metric)
            rec_metrics = evaluate_metrics(y_true, pred)["overall"]
            row = {
                "w_lowmid": 1.0,
                "w_high": float(w_high),
                "selected_metric": cfg.metric,
                "selected_score": score,
                **rec_metrics,
            }
            records.append(row)
            if score < best_score:
                best_score = score
                best_w_lowmid = 1.0
                best_w_high = float(w_high)
    elif cfg.search_mode == "two_dim":
        # 为了速度，先展平，避免每次重复构造多维索引。
        yt = y_true.reshape(-1)
        low = y_lowmid_pred.reshape(-1)
        high = y_high_pred.reshape(-1)
        y_mean = float(np.mean(yt))
        sst = float(np.sum((yt - y_mean) ** 2))

        for w_lowmid in weights:
            base = float(w_lowmid) * low
            for w_high in weights:
                yp = base + float(w_high) * high
                err = yp - yt
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                r2 = float(1.0 - np.sum(err ** 2) / sst) if sst > 0 else float("nan")
                score = rmse if cfg.metric == "rmse" else mae
                records.append({
                    "w_lowmid": float(w_lowmid),
                    "w_high": float(w_high),
                    "selected_metric": cfg.metric,
                    "selected_score": score,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "samples": int(y_true.size),
                })
                if score < best_score:
                    best_score = score
                    best_w_lowmid = float(w_lowmid)
                    best_w_high = float(w_high)
    else:
        raise ValueError(f"未知 search_mode: {cfg.search_mode}")

    result_df = pd.DataFrame(records).sort_values("selected_score", ascending=True).reset_index(drop=True)
    return best_w_lowmid, best_w_high, result_df


# =========================================================
# 4. 保存和绘图
# =========================================================
def save_prediction_csv(
    path: Path,
    y_true_scaled: np.ndarray,
    y_add_scaled: np.ndarray,
    y_fixed_scaled: np.ndarray,
    y_true_original: Optional[np.ndarray],
    y_add_original: Optional[np.ndarray],
    y_fixed_original: Optional[np.ndarray],
    y_lowmid_pred: np.ndarray,
    y_high_pred: np.ndarray,
    w_lowmid: float,
    w_high: float,
) -> None:
    rows = []
    n, farms = y_true_scaled.shape
    for sample_idx in range(n):
        for farm_idx in range(farms):
            row = {
                "sample_idx": sample_idx,
                "farm_id": farm_idx + 1,
                "w_lowmid": float(w_lowmid),
                "w_high": float(w_high),
                "y_true_scaled": float(y_true_scaled[sample_idx, farm_idx]),
                "y_pred_add_scaled": float(y_add_scaled[sample_idx, farm_idx]),
                "y_pred_fixed_scaled": float(y_fixed_scaled[sample_idx, farm_idx]),
                "error_add_scaled": float(y_add_scaled[sample_idx, farm_idx] - y_true_scaled[sample_idx, farm_idx]),
                "error_fixed_scaled": float(y_fixed_scaled[sample_idx, farm_idx] - y_true_scaled[sample_idx, farm_idx]),
                "abs_error_add_scaled": float(abs(y_add_scaled[sample_idx, farm_idx] - y_true_scaled[sample_idx, farm_idx])),
                "abs_error_fixed_scaled": float(abs(y_fixed_scaled[sample_idx, farm_idx] - y_true_scaled[sample_idx, farm_idx])),
                "lowmid_pred_scaled": float(y_lowmid_pred[sample_idx, farm_idx]),
                "high_pred_scaled": float(y_high_pred[sample_idx, farm_idx]),
            }
            if y_true_original is not None and y_add_original is not None and y_fixed_original is not None:
                row.update({
                    "y_true": float(y_true_original[sample_idx, farm_idx]),
                    "y_pred_add": float(y_add_original[sample_idx, farm_idx]),
                    "y_pred_fixed": float(y_fixed_original[sample_idx, farm_idx]),
                    "error_add": float(y_add_original[sample_idx, farm_idx] - y_true_original[sample_idx, farm_idx]),
                    "error_fixed": float(y_fixed_original[sample_idx, farm_idx] - y_true_original[sample_idx, farm_idx]),
                    "abs_error_add": float(abs(y_add_original[sample_idx, farm_idx] - y_true_original[sample_idx, farm_idx])),
                    "abs_error_fixed": float(abs(y_fixed_original[sample_idx, farm_idx] - y_true_original[sample_idx, farm_idx])),
                })
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def plot_prediction_curves(
    fig_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_add: np.ndarray,
    y_fixed: np.ndarray,
    max_points: int,
    y_label: str,
) -> None:
    show_points = min(max_points, len(y_true))
    for farm_idx in range(y_true.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:show_points, farm_idx], label="true")
        plt.plot(y_add[:show_points, farm_idx], label="add_fusion")
        plt.plot(y_fixed[:show_points, farm_idx], label="fixed_weight_fusion")
        plt.xlabel("Sample")
        plt.ylabel(y_label)
        plt.title(f"Fixed-weight Fusion Prediction - {split} - Farm {farm_idx + 1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{split}_farm_{farm_idx + 1}_fixed_weight_fusion_prediction.png", dpi=220)
        plt.close()


def plot_component_curves(
    fig_dir: Path,
    split: str,
    y_lowmid_pred: np.ndarray,
    y_high_pred: np.ndarray,
    w_lowmid: float,
    w_high: float,
    max_points: int,
) -> None:
    show_points = min(max_points, len(y_lowmid_pred))
    for farm_idx in range(y_lowmid_pred.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(y_lowmid_pred[:show_points, farm_idx], label="lowmid_gat_pred")
        plt.plot(y_high_pred[:show_points, farm_idx], label="high_resnet_pred")
        plt.plot(w_lowmid * y_lowmid_pred[:show_points, farm_idx], label="weighted_lowmid")
        plt.plot(w_high * y_high_pred[:show_points, farm_idx], label="weighted_high")
        plt.xlabel("Sample")
        plt.ylabel("Scaled component")
        plt.title(f"Fixed-weight Fusion Components - {split} - Farm {farm_idx + 1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{split}_farm_{farm_idx + 1}_fixed_weight_components.png", dpi=220)
        plt.close()


def plot_error_curve(
    fig_dir: Path,
    split: str,
    y_true: np.ndarray,
    y_add: np.ndarray,
    y_fixed: np.ndarray,
    max_points: int,
) -> None:
    show_points = min(max_points, len(y_true))
    add_abs_err = np.abs(y_add - y_true).mean(axis=1)
    fixed_abs_err = np.abs(y_fixed - y_true).mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(add_abs_err[:show_points], label="add_fusion_mean_abs_error")
    plt.plot(fixed_abs_err[:show_points], label="fixed_weight_mean_abs_error")
    plt.xlabel("Sample")
    plt.ylabel("Mean absolute error")
    plt.title(f"Fusion Error Curve - {split}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split}_fixed_weight_fusion_error_curve.png", dpi=220)
    plt.close()


def plot_weight_search_heatmap(search_df: pd.DataFrame, fig_dir: Path, cfg: Config) -> None:
    if cfg.search_mode != "two_dim":
        plt.figure(figsize=(8, 4))
        plt.plot(search_df["w_high"], search_df["selected_score"])
        plt.xlabel("w_high")
        plt.ylabel(cfg.metric.upper())
        plt.title("Fixed-weight Search Curve")
        plt.tight_layout()
        plt.savefig(fig_dir / "fixed_weight_search_curve.png", dpi=220)
        plt.close()
        return

    pivot = search_df.pivot_table(index="w_lowmid", columns="w_high", values="selected_score")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=cfg.metric.upper())
    x_ticks = np.linspace(0, pivot.shape[1] - 1, min(6, pivot.shape[1])).astype(int)
    y_ticks = np.linspace(0, pivot.shape[0] - 1, min(6, pivot.shape[0])).astype(int)
    plt.xticks(x_ticks, [f"{pivot.columns[i]:.2f}" for i in x_ticks])
    plt.yticks(y_ticks, [f"{pivot.index[i]:.2f}" for i in y_ticks])
    plt.xlabel("w_high")
    plt.ylabel("w_lowmid")
    plt.title("Fixed-weight Search Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "fixed_weight_search_heatmap.png", dpi=220)
    plt.close()


# =========================================================
# 5. 主流程
# =========================================================
def main() -> None:
    cfg = parse_args()
    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)
    save_json(asdict(cfg), run_dir / "metrics" / "config.json")

    print("===== Fixed-weight Fusion: GAT Low/Mid + High ResNet =====")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    search_data = load_branch_data(cfg, cfg.weight_search_split)
    w_lowmid, w_high, search_df = search_fixed_weights(
        y_true=search_data["y_final_true_scaled"],
        y_lowmid_pred=search_data["y_lowmid_pred"],
        y_high_pred=search_data["y_high_pred"],
        cfg=cfg,
    )
    search_df.to_csv(run_dir / "metrics" / "fixed_weight_search_results.csv", index=False, encoding="utf-8-sig")
    search_df.head(50).to_csv(run_dir / "metrics" / "fixed_weight_search_top50.csv", index=False, encoding="utf-8-sig")
    plot_weight_search_heatmap(search_df, run_dir / "figures", cfg)

    best_info = {
        "weight_search_split": cfg.weight_search_split,
        "search_mode": cfg.search_mode,
        "selected_metric": cfg.metric,
        "w_lowmid": float(w_lowmid),
        "w_high": float(w_high),
        "best_selected_score": float(search_df.iloc[0]["selected_score"]),
        "best_mae": float(search_df.iloc[0]["mae"]),
        "best_rmse": float(search_df.iloc[0]["rmse"]),
        "best_r2": float(search_df.iloc[0]["r2"]),
    }
    save_json(best_info, run_dir / "metrics" / "best_fixed_weights.json")

    print("\n===== Best fixed weights on search split =====")
    print(json.dumps(best_info, ensure_ascii=False, indent=2))

    eval_data = load_branch_data(cfg, cfg.eval_split)
    y_lowmid_pred = eval_data["y_lowmid_pred"]
    y_high_pred = eval_data["y_high_pred"]
    y_true_scaled = eval_data["y_final_true_scaled"]

    y_add_scaled = y_lowmid_pred + y_high_pred
    y_fixed_scaled = float(w_lowmid) * y_lowmid_pred + float(w_high) * y_high_pred

    # 保存 scaled 数组。
    np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_lowmid_pred_scaled.npy", y_lowmid_pred)
    np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_high_pred_scaled.npy", y_high_pred)
    np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_true_scaled.npy", y_true_scaled)
    np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_add_scaled.npy", y_add_scaled)
    np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_fixed_weight_scaled.npy", y_fixed_scaled)
    np.save(run_dir / "arrays" / "best_fixed_weights.npy", np.array([w_lowmid, w_high], dtype=np.float32))

    # scaled 指标。
    metrics_scaled = {
        "split": cfg.eval_split,
        "scale_domain": "scaled",
        "true_source": eval_data["true_source"],
        "best_fixed_weights": best_info,
        "add_fusion": evaluate_metrics(y_true_scaled, y_add_scaled),
        "fixed_weight_fusion": evaluate_metrics(y_true_scaled, y_fixed_scaled),
    }
    save_json(metrics_scaled, run_dir / "metrics" / f"{cfg.eval_split}_metrics_scaled.json")

    # 反标准化。
    scaler_mean, scaler_scale, scaler_source = load_scaler(Path(cfg.data_dir))
    y_true_original = inverse_scale(y_true_scaled, scaler_mean, scaler_scale)
    y_add_original = inverse_scale(y_add_scaled, scaler_mean, scaler_scale)
    y_fixed_original = inverse_scale(y_fixed_scaled, scaler_mean, scaler_scale)

    metrics_original = None
    if y_true_original is not None and y_add_original is not None and y_fixed_original is not None:
        np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_true.npy", y_true_original)
        np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_add.npy", y_add_original)
        np.save(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_fixed_weight.npy", y_fixed_original)
        metrics_original = {
            "split": cfg.eval_split,
            "scale_domain": "original_power",
            "scaler_source": scaler_source,
            "true_source": eval_data["true_source"],
            "best_fixed_weights": best_info,
            "add_fusion": evaluate_metrics(y_true_original, y_add_original),
            "fixed_weight_fusion": evaluate_metrics(y_true_original, y_fixed_original),
        }
        save_json(metrics_original, run_dir / "metrics" / f"{cfg.eval_split}_metrics.json")

    # CSV 明细。
    save_prediction_csv(
        run_dir / "predictions" / f"{cfg.eval_split}_predictions.csv",
        y_true_scaled=y_true_scaled,
        y_add_scaled=y_add_scaled,
        y_fixed_scaled=y_fixed_scaled,
        y_true_original=y_true_original,
        y_add_original=y_add_original,
        y_fixed_original=y_fixed_original,
        y_lowmid_pred=y_lowmid_pred,
        y_high_pred=y_high_pred,
        w_lowmid=w_lowmid,
        w_high=w_high,
    )

    # 绘图：优先原始尺度，否则 scaled。
    if y_true_original is not None and y_add_original is not None and y_fixed_original is not None:
        plot_prediction_curves(
            run_dir / "figures", cfg.eval_split,
            y_true=y_true_original,
            y_add=y_add_original,
            y_fixed=y_fixed_original,
            max_points=cfg.max_plot_points,
            y_label="Power",
        )
        plot_error_curve(run_dir / "figures", cfg.eval_split, y_true_original, y_add_original, y_fixed_original, cfg.max_plot_points)
    else:
        plot_prediction_curves(
            run_dir / "figures", cfg.eval_split,
            y_true=y_true_scaled,
            y_add=y_add_scaled,
            y_fixed=y_fixed_scaled,
            max_points=cfg.max_plot_points,
            y_label="Scaled power",
        )
        plot_error_curve(run_dir / "figures", cfg.eval_split, y_true_scaled, y_add_scaled, y_fixed_scaled, cfg.max_plot_points)

    plot_component_curves(
        run_dir / "figures", cfg.eval_split,
        y_lowmid_pred=y_lowmid_pred,
        y_high_pred=y_high_pred,
        w_lowmid=w_lowmid,
        w_high=w_high,
        max_points=cfg.max_plot_points,
    )

    overview = {
        "task": "Fixed-weight fusion: GAT low/mid branch + High ResNet branch",
        "run_dir": str(run_dir),
        "weight_search_split": cfg.weight_search_split,
        "eval_split": cfg.eval_split,
        "lowmid_pred_path": eval_data["lowmid_pred_path"],
        "lowmid_true_path": eval_data["lowmid_true_path"],
        "high_pred_path": eval_data["high_pred_path"],
        "high_true_path": eval_data["high_true_path"],
        "true_source": eval_data["true_source"],
        "scaler_source": scaler_source,
        "best_fixed_weights": best_info,
        "shapes": {
            "y_lowmid_pred": list(y_lowmid_pred.shape),
            "y_high_pred": list(y_high_pred.shape),
            "y_final_pred_add_scaled": list(y_add_scaled.shape),
            "y_final_pred_fixed_weight_scaled": list(y_fixed_scaled.shape),
            "y_final_true_scaled": list(y_true_scaled.shape),
        },
        "final_metrics_scaled": {
            "add_fusion": metrics_scaled["add_fusion"]["overall"],
            "fixed_weight_fusion": metrics_scaled["fixed_weight_fusion"]["overall"],
        },
        "final_metrics_original": {
            "add_fusion": metrics_original["add_fusion"]["overall"] if metrics_original is not None else None,
            "fixed_weight_fusion": metrics_original["fixed_weight_fusion"]["overall"] if metrics_original is not None else None,
        },
        "output_files": {
            "best_fixed_weights": str(run_dir / "metrics" / "best_fixed_weights.json"),
            "search_results": str(run_dir / "metrics" / "fixed_weight_search_results.csv"),
            "metrics_scaled": str(run_dir / "metrics" / f"{cfg.eval_split}_metrics_scaled.json"),
            "metrics_original": str(run_dir / "metrics" / f"{cfg.eval_split}_metrics.json") if metrics_original is not None else None,
            "predictions_csv": str(run_dir / "predictions" / f"{cfg.eval_split}_predictions.csv"),
            "y_final_pred_add_scaled": str(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_add_scaled.npy"),
            "y_final_pred_fixed_weight_scaled": str(run_dir / "arrays" / f"{cfg.eval_split}_y_final_pred_fixed_weight_scaled.npy"),
        },
    }
    save_json(overview, run_dir / "metrics" / "overview.json")

    print("\n===== 固定加权融合完成 =====")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
