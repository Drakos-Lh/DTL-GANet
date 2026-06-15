"""
零件匹配模块升级实验脚本

目标：
1. 扫描不同零件尺寸 edge_count，回答“4个节点3条边是否最优”；
2. 构筑零件匹配 KNN 预测 baseline，用 MAE/RMSE 评价最优；
3. 对方向、长度形状、总时长、斜率形状等图结构特征做重要性排序；
4. 自动输出表格和可视化图片。

使用前提：
- 每个 edge_count 对应一个零件库目录：outputs/lowmid_piece_graphs_E{edge_count}/{split}/farm_{i}_pieces.csv
- 如果 edge_count=3 且上述目录不存在，会自动回退到你现有的 outputs/lowmid_piece_graphs
- vmd_data 下存在 train/val/test_y_lowmid.npy，shape 通常为 (N, 1, 6)

推荐运行：
python piece_matching_full_sweep_E1_E6_save_visdata.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 1. 配置区：优先只改这里
# =========================================================
@dataclass
class Config:
    # 不同零件尺寸的目录模板。例：E3 表示 4 个节点、3 条边。
    piece_root_template: str = "outputs/lowmid_piece_graphs_E{edge_count}"
    fallback_piece_root_e3: str = "outputs/lowmid_piece_graphs"

    data_dir: str = "vmd_data"
    output_root: str = "outputs"
    run_name: str = "piece_matching_full_sweep_E1_E6"

    # 需要扫描的零件边数。edge_count=3 就是当前的“4个节点3条线”。
    edge_counts: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    # 用训练集零件库匹配验证集，避免把测试集未来信息用于匹配。
    candidate_split: str = "train"
    target_split: str = "val"

    num_farms: int = 6
    top_k: int = 5

    # 默认只做跨风电场匹配；如果想允许同一风电场历史片段，也可以改成 False。
    exclude_same_farm: bool = True

    # 内存安全设置
    target_block_size: int = 256
    candidate_block_size: int = 4096
    max_pieces_per_source: int | None = None
    random_seed: int = 42

    # 基础相似度权重，会在不同特征组合下自动归一化
    weight_direction: float = 0.45
    weight_length_shape: float = 0.25
    weight_total_duration: float = 0.10
    weight_slope_shape: float = 0.20

    # 临界点判定：选择“RMSE 距离最优不超过 1% 的最小零件尺寸”
    elbow_tolerance: float = 0.01

    # 输出 Top-k 明细可能很大；False 时只保存汇总表，速度更快、占用更低。
    save_match_details: bool = True


CFG = Config()
DIR_MAP = {"up": 1.0, "down": -1.0, "flat": 0.0}

BASE_FEATURE_WEIGHTS = {
    "direction": "weight_direction",
    "length_shape": "weight_length_shape",
    "total_duration": "weight_total_duration",
    "slope_shape": "weight_slope_shape",
}

FEATURE_SETS = {
    "all": ("direction", "length_shape", "total_duration", "slope_shape"),
    "no_direction": ("length_shape", "total_duration", "slope_shape"),
    "no_length_shape": ("direction", "total_duration", "slope_shape"),
    "no_total_duration": ("direction", "length_shape", "slope_shape"),
    "no_slope_shape": ("direction", "length_shape", "total_duration"),
    "direction_only": ("direction",),
    "length_shape_only": ("length_shape",),
    "total_duration_only": ("total_duration",),
    "slope_shape_only": ("slope_shape",),
}


# =========================================================
# 2. 基础工具
# =========================================================
def ensure_dirs(run_dir: Path) -> None:
    for name in ["metrics", "tables", "figures", "match_details", "vis_data"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
    }


def resolve_piece_root(cfg: Config, edge_count: int) -> Path | None:
    root = Path(cfg.piece_root_template.format(edge_count=edge_count))
    if root.exists():
        return root
    if edge_count == 3:
        fallback = Path(cfg.fallback_piece_root_e3)
        if fallback.exists():
            return fallback
    return None


def required_columns(edge_count: int) -> List[str]:
    cols = ["split", "farm_id", "sample_idx", "piece_local_id", "start_time_idx", "end_time_idx"]
    for e in range(edge_count):
        cols += [f"edge_{e}_direction", f"edge_{e}_length", f"edge_{e}_avg_slope"]
    return cols


def load_y(data_dir: Path, split: str) -> np.ndarray:
    path = data_dir / f"{split}_y_lowmid.npy"
    if not path.exists():
        raise FileNotFoundError(f"未找到目标文件: {path}")
    y = np.load(path)
    if y.ndim == 3:
        y = y[:, 0, :]
    if y.ndim != 2:
        raise ValueError(f"{path} 的 shape 应为 (N, 1, farms) 或 (N, farms)，实际为 {y.shape}")
    return y.astype(np.float32)


def load_piece_tables(piece_root: Path, split: str, num_farms: int, edge_count: int) -> Dict[int, pd.DataFrame]:
    tables: Dict[int, pd.DataFrame] = {}
    need_cols = required_columns(edge_count)
    for farm_id in range(1, num_farms + 1):
        csv_path = piece_root / split / f"farm_{farm_id}_pieces.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到零件文件: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少列: {missing}")
        tables[farm_id] = df[need_cols].copy()
    return tables


def maybe_subsample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.max_pieces_per_source is None or len(df) <= cfg.max_pieces_per_source:
        return df.reset_index(drop=True)
    return df.sample(cfg.max_pieces_per_source, random_state=cfg.random_seed).reset_index(drop=True)


# =========================================================
# 3. 动态 edge_count 特征构建
# =========================================================
def direction_vector(df: pd.DataFrame, edge_count: int) -> np.ndarray:
    return np.stack(
        [df[f"edge_{e}_direction"].map(DIR_MAP).to_numpy(dtype=np.float32) for e in range(edge_count)],
        axis=1,
    )


def length_vector(df: pd.DataFrame, edge_count: int) -> np.ndarray:
    return np.stack(
        [df[f"edge_{e}_length"].to_numpy(dtype=np.float32) for e in range(edge_count)],
        axis=1,
    )


def slope_vector(df: pd.DataFrame, edge_count: int) -> np.ndarray:
    return np.stack(
        [df[f"edge_{e}_avg_slope"].to_numpy(dtype=np.float32) for e in range(edge_count)],
        axis=1,
    )


def normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.sum(np.abs(arr), axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def build_feature_bank(df: pd.DataFrame, edge_count: int, y_split: np.ndarray) -> pd.DataFrame:
    bank = df.copy().reset_index(drop=True)

    dirs = direction_vector(bank, edge_count)
    lens = length_vector(bank, edge_count)
    slopes = slope_vector(bank, edge_count)

    lens_norm = normalize_rows(lens)
    slope_mag_norm = normalize_rows(np.abs(slopes))

    for e in range(edge_count):
        bank[f"dir_code_{e}"] = dirs[:, e]
        bank[f"len_norm_{e}"] = lens_norm[:, e]
        bank[f"slope_mag_norm_{e}"] = slope_mag_norm[:, e]

    bank["total_duration"] = lens.sum(axis=1)

    # 将每个零件对应的下一步真实 lowmid 作为 KNN baseline 的可取标签。
    ys = []
    valid = []
    for _, row in bank.iterrows():
        sample_idx = int(row["sample_idx"])
        farm_idx = int(row["farm_id"]) - 1
        ok = 0 <= sample_idx < len(y_split) and 0 <= farm_idx < y_split.shape[1]
        valid.append(ok)
        ys.append(float(y_split[sample_idx, farm_idx]) if ok else np.nan)
    bank["future_y"] = ys
    bank = bank[np.array(valid, dtype=bool)].reset_index(drop=True)
    return bank


def feature_arrays(df: pd.DataFrame, edge_count: int) -> Dict[str, np.ndarray]:
    return {
        "dir": df[[f"dir_code_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "len": df[[f"len_norm_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "total": df[["total_duration"]].to_numpy(dtype=np.float32),
        "slope": df[[f"slope_mag_norm_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "future_y": df[["future_y"]].to_numpy(dtype=np.float32),
    }


def normalized_weights(cfg: Config, feature_set: Iterable[str]) -> Dict[str, float]:
    raw = {name: float(getattr(cfg, BASE_FEATURE_WEIGHTS[name])) for name in feature_set}
    s = sum(raw.values())
    if s <= 0:
        raise ValueError(f"特征权重之和必须大于0: {raw}")
    return {k: v / s for k, v in raw.items()}


def compute_similarity_block(
    target: Dict[str, np.ndarray],
    cand: Dict[str, np.ndarray],
    cfg: Config,
    feature_set: Tuple[str, ...],
) -> np.ndarray:
    weights = normalized_weights(cfg, feature_set)
    score = None

    if "direction" in feature_set:
        direction_match = (target["dir"][:, None, :] == cand["dir"][None, :, :]).mean(axis=2, dtype=np.float32)
        score = weights["direction"] * direction_match if score is None else score + weights["direction"] * direction_match

    if "length_shape" in feature_set:
        len_l1 = np.abs(target["len"][:, None, :] - cand["len"][None, :, :]).mean(axis=2, dtype=np.float32)
        len_shape_sim = 1.0 - np.clip(len_l1, 0.0, 1.0)
        score = weights["length_shape"] * len_shape_sim if score is None else score + weights["length_shape"] * len_shape_sim

    if "total_duration" in feature_set:
        total_diff = np.abs(target["total"] - cand["total"].T)
        denom = np.maximum(np.maximum(target["total"], cand["total"].T), 1.0)
        total_duration_sim = 1.0 - np.clip(total_diff / denom, 0.0, 1.0)
        score = weights["total_duration"] * total_duration_sim if score is None else score + weights["total_duration"] * total_duration_sim

    if "slope_shape" in feature_set:
        slope_l1 = np.abs(target["slope"][:, None, :] - cand["slope"][None, :, :]).mean(axis=2, dtype=np.float32)
        slope_shape_sim = 1.0 - np.clip(slope_l1, 0.0, 1.0)
        score = weights["slope_shape"] * slope_shape_sim if score is None else score + weights["slope_shape"] * slope_shape_sim

    if score is None:
        raise ValueError("feature_set 不能为空")
    return score.astype(np.float32)


# =========================================================
# 4. 零件匹配 KNN baseline
# =========================================================
def streaming_topk_predict_for_block(
    target_block_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    edge_count: int,
    cfg: Config,
    feature_set: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = feature_arrays(target_block_df, edge_count)
    n_t = len(target_block_df)
    k = min(cfg.top_k, len(cand_df))

    best_scores = np.full((n_t, k), -np.inf, dtype=np.float32)
    best_y = np.full((n_t, k), np.nan, dtype=np.float32)
    best_src_farm = np.full((n_t, k), -1, dtype=np.int32)
    best_src_sample = np.full((n_t, k), -1, dtype=np.int32)

    for c_start in range(0, len(cand_df), cfg.candidate_block_size):
        c_end = min(c_start + cfg.candidate_block_size, len(cand_df))
        cand_block_df = cand_df.iloc[c_start:c_end].reset_index(drop=True)
        cand = feature_arrays(cand_block_df, edge_count)
        sim = compute_similarity_block(target, cand, cfg, feature_set)

        cand_y = cand["future_y"].reshape(1, -1).repeat(n_t, axis=0)
        cand_src_farm = cand_block_df["farm_id"].to_numpy(dtype=np.int32).reshape(1, -1).repeat(n_t, axis=0)
        cand_src_sample = cand_block_df["sample_idx"].to_numpy(dtype=np.int32).reshape(1, -1).repeat(n_t, axis=0)

        merged_scores = np.concatenate([best_scores, sim], axis=1)
        merged_y = np.concatenate([best_y, cand_y], axis=1)
        merged_src_farm = np.concatenate([best_src_farm, cand_src_farm], axis=1)
        merged_src_sample = np.concatenate([best_src_sample, cand_src_sample], axis=1)

        part = np.argpartition(-merged_scores, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(n_t)[:, None]
        best_scores = merged_scores[row_idx, part]
        best_y = merged_y[row_idx, part]
        best_src_farm = merged_src_farm[row_idx, part]
        best_src_sample = merged_src_sample[row_idx, part]

        order = np.argsort(-best_scores, axis=1)
        best_scores = best_scores[row_idx, order]
        best_y = best_y[row_idx, order]
        best_src_farm = best_src_farm[row_idx, order]
        best_src_sample = best_src_sample[row_idx, order]

    # 用相似度加权平均作为预测值。
    weights = np.maximum(best_scores, 1e-6)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    pred = np.sum(weights * best_y, axis=1)

    return pred.astype(np.float32), best_scores, best_src_farm, best_src_sample


def evaluate_one_setting(
    target_tables: Dict[int, pd.DataFrame],
    candidate_tables: Dict[int, pd.DataFrame],
    y_target: np.ndarray,
    edge_count: int,
    cfg: Config,
    feature_set_name: str,
    run_dir: Path,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    feature_set = FEATURE_SETS[feature_set_name]
    rows = []
    detail_rows = []

    for target_farm in range(1, cfg.num_farms + 1):
        target_df = target_tables[target_farm].reset_index(drop=True)

        candidate_frames = []
        for source_farm in range(1, cfg.num_farms + 1):
            if cfg.exclude_same_farm and source_farm == target_farm:
                continue
            candidate_frames.append(maybe_subsample(candidate_tables[source_farm], cfg))

        if not candidate_frames:
            raise ValueError("候选零件库为空，请检查 exclude_same_farm 或数据目录。")

        cand_df = pd.concat(candidate_frames, axis=0, ignore_index=True)

        for t_start in range(0, len(target_df), cfg.target_block_size):
            t_end = min(t_start + cfg.target_block_size, len(target_df))
            block = target_df.iloc[t_start:t_end].reset_index(drop=True)
            pred_piece, best_scores, best_src_farm, best_src_sample = streaming_topk_predict_for_block(
                block, cand_df, edge_count, cfg, feature_set
            )

            for i_local, (_, trow) in enumerate(block.iterrows()):
                sample_idx = int(trow["sample_idx"])
                if sample_idx < 0 or sample_idx >= len(y_target):
                    continue
                y_true = float(y_target[sample_idx, target_farm - 1])
                rows.append({
                    "edge_count": edge_count,
                    "node_count": edge_count + 1,
                    "feature_set": feature_set_name,
                    "target_farm": target_farm,
                    "target_sample_idx": sample_idx,
                    "target_piece_local_id": int(trow["piece_local_id"]),
                    "y_true": y_true,
                    "y_pred_piece": float(pred_piece[i_local]),
                    "best_similarity": float(best_scores[i_local, 0]),
                    "best_source_farm": int(best_src_farm[i_local, 0]),
                    "best_source_sample_idx": int(best_src_sample[i_local, 0]),
                    "best_lag_samples": int(sample_idx - int(best_src_sample[i_local, 0])),
                })
                if cfg.save_match_details and feature_set_name == "all":
                    for rank in range(best_scores.shape[1]):
                        detail_rows.append({
                            "edge_count": edge_count,
                            "target_farm": target_farm,
                            "target_sample_idx": sample_idx,
                            "target_piece_local_id": int(trow["piece_local_id"]),
                            "rank": rank + 1,
                            "source_farm": int(best_src_farm[i_local, rank]),
                            "source_sample_idx": int(best_src_sample[i_local, rank]),
                            "lag_samples": int(sample_idx - int(best_src_sample[i_local, rank])),
                            "similarity": float(best_scores[i_local, rank]),
                        })

    piece_pred_df = pd.DataFrame(rows)
    if piece_pred_df.empty:
        raise ValueError(f"edge_count={edge_count}, feature_set={feature_set_name} 没有有效预测行。")

    # 一个 sample 可能拆出多个零件；这里对同一风电场同一样本的多个零件预测求平均。
    sample_pred_df = (
        piece_pred_df
        .groupby(["edge_count", "node_count", "feature_set", "target_farm", "target_sample_idx"], as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred_piece", "mean"),
            avg_best_similarity=("best_similarity", "mean"),
        )
    )

    sample_pred_df["abs_error"] = np.abs(sample_pred_df["y_true"] - sample_pred_df["y_pred"])
    sample_pred_df["squared_error"] = (sample_pred_df["y_true"] - sample_pred_df["y_pred"]) ** 2

    overall = evaluate_metrics(sample_pred_df["y_true"].to_numpy(), sample_pred_df["y_pred"].to_numpy())

    per_farm = []
    for farm_id, sub in sample_pred_df.groupby("target_farm"):
        m = evaluate_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        per_farm.append({
            "edge_count": edge_count,
            "node_count": edge_count + 1,
            "feature_set": feature_set_name,
            "target_farm": int(farm_id),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "samples": int(len(sub)),
        })

    metrics = {
        "edge_count": edge_count,
        "node_count": edge_count + 1,
        "feature_set": feature_set_name,
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "num_sample_predictions": int(len(sample_pred_df)),
        "num_piece_predictions": int(len(piece_pred_df)),
    }


    # ---------- 保存后续可视化需要的紧凑中间数据 ----------
    # 1) 每个目标零件的 rank-1 匹配结果：用于画时滞分布、source farm 热力图、案例图。
    if feature_set_name == "all":
        vis_piece_cols = [
            "edge_count", "node_count", "feature_set",
            "target_farm", "target_sample_idx", "target_piece_local_id",
            "y_true", "y_pred_piece", "best_similarity",
            "best_source_farm", "best_source_sample_idx", "best_lag_samples",
        ]
        piece_pred_df[vis_piece_cols].to_csv(
            run_dir / "vis_data" / f"piece_best_match_edge_{edge_count}_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 2) source farm 统计：用于画 target_farm × source_farm 热力图。
        source_summary = (
            piece_pred_df
            .groupby(["edge_count", "node_count", "target_farm", "best_source_farm"], as_index=False)
            .agg(
                matched_count=("best_similarity", "count"),
                avg_similarity=("best_similarity", "mean"),
                max_similarity=("best_similarity", "max"),
                avg_abs_error=("y_pred_piece", lambda x: float(np.nan)),
            )
        )
        # avg_abs_error 在 source 粒度上单独计算，避免 groupby lambda 拿不到 y_true。
        tmp = piece_pred_df.copy()
        tmp["abs_error_piece"] = np.abs(tmp["y_true"] - tmp["y_pred_piece"])
        source_error = (
            tmp.groupby(["edge_count", "node_count", "target_farm", "best_source_farm"], as_index=False)
            .agg(avg_abs_error=("abs_error_piece", "mean"))
        )
        source_summary = source_summary.drop(columns=["avg_abs_error"]).merge(
            source_error,
            on=["edge_count", "node_count", "target_farm", "best_source_farm"],
            how="left",
        )
        source_summary.to_csv(
            run_dir / "vis_data" / f"source_farm_summary_edge_{edge_count}_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 3) 时滞分布：用于画 lag histogram / lag heatmap。
        lag_summary = (
            piece_pred_df
            .groupby(["edge_count", "node_count", "target_farm", "best_source_farm", "best_lag_samples"], as_index=False)
            .agg(
                lag_count=("best_similarity", "count"),
                avg_similarity=("best_similarity", "mean"),
            )
            .sort_values(["target_farm", "best_source_farm", "best_lag_samples"])
        )
        lag_summary.to_csv(
            run_dir / "vis_data" / f"lag_distribution_edge_{edge_count}_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

    detail_df = pd.DataFrame(detail_rows)
    if cfg.save_match_details and feature_set_name == "all" and not detail_df.empty:
        detail_path = run_dir / "match_details" / f"edge_{edge_count}_all_topk_details.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    per_farm_df = pd.DataFrame(per_farm)
    return metrics, per_farm_df, sample_pred_df


def prepare_banks_for_edge(edge_count: int, cfg: Config) -> Tuple[Path, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], np.ndarray]:
    piece_root = resolve_piece_root(cfg, edge_count)
    if piece_root is None:
        raise FileNotFoundError(
            f"未找到 edge_count={edge_count} 的零件库。应存在 {cfg.piece_root_template.format(edge_count=edge_count)}"
        )

    data_dir = Path(cfg.data_dir)
    y_candidate = load_y(data_dir, cfg.candidate_split)
    y_target = load_y(data_dir, cfg.target_split)

    raw_candidate = load_piece_tables(piece_root, cfg.candidate_split, cfg.num_farms, edge_count)
    raw_target = load_piece_tables(piece_root, cfg.target_split, cfg.num_farms, edge_count)

    candidate_tables = {fid: build_feature_bank(df, edge_count, y_candidate) for fid, df in raw_candidate.items()}
    target_tables = {fid: build_feature_bank(df, edge_count, y_target) for fid, df in raw_target.items()}
    return piece_root, target_tables, candidate_tables, y_target


# =========================================================
# 5. 汇总、临界点、特征重要性和绘图
# =========================================================
def find_critical_point(summary_df: pd.DataFrame, cfg: Config) -> Dict:
    all_df = summary_df[summary_df["feature_set"] == "all"].copy()
    all_df = all_df.sort_values("edge_count")
    best_row = all_df.loc[all_df["rmse"].idxmin()].to_dict()
    threshold = float(best_row["rmse"]) * (1.0 + cfg.elbow_tolerance)
    critical_row = all_df[all_df["rmse"] <= threshold].sort_values("edge_count").iloc[0].to_dict()
    return {
        "best_by_rmse": best_row,
        "critical_point_by_elbow": critical_row,
        "rule": f"选择 RMSE 不超过最优 RMSE 的 {cfg.elbow_tolerance * 100:.1f}% 的最小 edge_count",
    }


def build_feature_importance(summary_df: pd.DataFrame, chosen_edge_count: int) -> pd.DataFrame:
    sub = summary_df[summary_df["edge_count"] == chosen_edge_count].copy()
    all_rmse = float(sub[sub["feature_set"] == "all"].iloc[0]["rmse"])

    rows = []
    removal_map = {
        "direction": "no_direction",
        "length_shape": "no_length_shape",
        "total_duration": "no_total_duration",
        "slope_shape": "no_slope_shape",
    }
    single_map = {
        "direction": "direction_only",
        "length_shape": "length_shape_only",
        "total_duration": "total_duration_only",
        "slope_shape": "slope_shape_only",
    }

    for feature, no_name in removal_map.items():
        no_rmse = float(sub[sub["feature_set"] == no_name].iloc[0]["rmse"])
        single_rmse = float(sub[sub["feature_set"] == single_map[feature]].iloc[0]["rmse"])
        rows.append({
            "edge_count": chosen_edge_count,
            "feature": feature,
            "all_rmse": all_rmse,
            "rmse_without_feature": no_rmse,
            "importance_by_removal": no_rmse - all_rmse,
            "single_feature_rmse": single_rmse,
        })

    out = pd.DataFrame(rows).sort_values("importance_by_removal", ascending=False).reset_index(drop=True)
    out["importance_rank"] = np.arange(1, len(out) + 1)
    return out


def export_visualization_tables(
    summary_df: pd.DataFrame,
    per_farm_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    critical: Dict,
    run_dir: Path,
) -> None:
    """
    额外导出后续论文/PPT可视化所需的干净数据表。
    这些表不依赖 matplotlib，可以之后用 Excel、Origin、Python 重新画图。
    """
    vis_dir = run_dir / "vis_data"
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_edge = int(critical["best_by_rmse"]["edge_count"])
    critical_edge = int(critical["critical_point_by_elbow"]["edge_count"])

    # 1) 尺寸扫描曲线数据：edge_count - MAE/RMSE。
    size_curve = (
        summary_df[summary_df["feature_set"] == "all"]
        .sort_values("edge_count")
        .reset_index(drop=True)
        .copy()
    )
    size_curve["is_best_by_rmse"] = size_curve["edge_count"].astype(int) == best_edge
    size_curve["is_critical_point"] = size_curve["edge_count"].astype(int) == critical_edge
    size_curve.to_csv(vis_dir / "size_sweep_curve_all_features.csv", index=False, encoding="utf-8-sig")

    # 2) 每个风电场的尺寸扫描数据：用于画分风场折线图。
    per_farm_curve = (
        per_farm_df[per_farm_df["feature_set"] == "all"]
        .sort_values(["target_farm", "edge_count"])
        .reset_index(drop=True)
        .copy()
    )
    per_farm_curve.to_csv(vis_dir / "per_farm_size_sweep_all_features.csv", index=False, encoding="utf-8-sig")

    # 3) 特征消融/重要性数据：用于柱状图。
    importance_df.to_csv(vis_dir / "feature_importance_for_barplot.csv", index=False, encoding="utf-8-sig")

    # 4) 尺寸 × 特征组合 RMSE 矩阵：用于热力图。
    rmse_matrix = summary_df.pivot_table(index="feature_set", columns="edge_count", values="rmse", aggfunc="mean")
    rmse_matrix.to_csv(vis_dir / "rmse_matrix_feature_set_by_edge_count.csv", encoding="utf-8-sig")

    mae_matrix = summary_df.pivot_table(index="feature_set", columns="edge_count", values="mae", aggfunc="mean")
    mae_matrix.to_csv(vis_dir / "mae_matrix_feature_set_by_edge_count.csv", encoding="utf-8-sig")

    # 5) 全部指标长表：用于后续任意自定义可视化。
    summary_df.to_csv(vis_dir / "all_metrics_long_table.csv", index=False, encoding="utf-8-sig")

    # 6) 最优点信息另存一份 CSV，方便论文表格引用。
    pd.DataFrame([
        {"type": "best_by_rmse", **critical["best_by_rmse"]},
        {"type": "critical_point_by_elbow", **critical["critical_point_by_elbow"]},
    ]).to_csv(vis_dir / "best_and_critical_point.csv", index=False, encoding="utf-8-sig")


def plot_size_sweep(summary_df: pd.DataFrame, run_dir: Path) -> None:
    df = summary_df[summary_df["feature_set"] == "all"].sort_values("edge_count")
    plt.figure(figsize=(8, 5))
    plt.plot(df["edge_count"], df["mae"], marker="o", label="MAE")
    plt.plot(df["edge_count"], df["rmse"], marker="o", label="RMSE")
    plt.xlabel("Edge count in one piece")
    plt.ylabel("Validation error")
    plt.title("Piece Size Sweep: Matching Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "piece_size_sweep_mae_rmse.png", dpi=200)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, run_dir: Path) -> None:
    df = importance_df.sort_values("importance_by_removal", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance_by_removal"])
    plt.xlabel("RMSE increase after removing feature")
    plt.ylabel("Graph feature")
    plt.title("Feature Importance by Ablation")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "feature_importance_ablation.png", dpi=200)
    plt.close()


def plot_ablation_heatmap(summary_df: pd.DataFrame, run_dir: Path) -> None:
    pivot = summary_df.pivot_table(index="feature_set", columns="edge_count", values="rmse", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="RMSE")
    plt.xlabel("Edge count")
    plt.ylabel("Feature set")
    plt.title("RMSE Heatmap: Piece Size × Feature Set")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "size_feature_ablation_heatmap.png", dpi=200)
    plt.close()


def plot_source_ranking_from_details(run_dir: Path, chosen_edge_count: int, cfg: Config) -> None:
    detail_path = run_dir / "match_details" / f"edge_{chosen_edge_count}_all_topk_details.csv"
    summary_path = run_dir / "vis_data" / f"source_farm_summary_edge_{chosen_edge_count}_all.csv"

    if detail_path.exists():
        df = pd.read_csv(detail_path)
        if df.empty:
            return
        agg = (
            df.groupby(["target_farm", "source_farm"], as_index=False)
            .agg(avg_similarity=("similarity", "mean"), matched_count=("similarity", "count"))
        )
    elif summary_path.exists():
        agg = pd.read_csv(summary_path).rename(columns={"best_source_farm": "source_farm"})
        if agg.empty:
            return
    else:
        return

    agg.to_csv(run_dir / "tables" / "source_farm_ranking_at_critical_edge.csv", index=False, encoding="utf-8-sig")

    mat = np.full((cfg.num_farms, cfg.num_farms), np.nan, dtype=np.float32)
    for _, row in agg.iterrows():
        mat[int(row["target_farm"]) - 1, int(row["source_farm"]) - 1] = float(row["avg_similarity"])

    plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Average similarity")
    plt.xticks(range(cfg.num_farms), [f"S{i}" for i in range(1, cfg.num_farms + 1)])
    plt.yticks(range(cfg.num_farms), [f"T{i}" for i in range(1, cfg.num_farms + 1)])
    plt.xlabel("Source farm")
    plt.ylabel("Target farm")
    plt.title(f"Source Farm Ranking Heatmap, edge={chosen_edge_count}")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "source_farm_similarity_heatmap.png", dpi=200)
    plt.close()


# =========================================================
# 6. 主流程
# =========================================================
def main() -> None:
    cfg = CFG
    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)
    save_json(asdict(cfg), run_dir / "metrics" / "config.json")

    all_metrics = []
    all_per_farm = []

    for edge_count in cfg.edge_counts:
        try:
            piece_root, target_tables, candidate_tables, y_target = prepare_banks_for_edge(edge_count, cfg)
        except Exception as e:
            print(f"[跳过] edge_count={edge_count}: {e}")
            continue

        print(f"\n===== edge_count={edge_count}, piece_root={piece_root} =====")
        piece_counts = {
            "edge_count": edge_count,
            "candidate_counts": {f"farm_{fid}": int(len(df)) for fid, df in candidate_tables.items()},
            "target_counts": {f"farm_{fid}": int(len(df)) for fid, df in target_tables.items()},
        }
        save_json(piece_counts, run_dir / "metrics" / f"piece_counts_edge_{edge_count}.json")

        for feature_set_name in FEATURE_SETS.keys():
            print(f"  -> feature_set={feature_set_name}")
            metrics, per_farm_df, sample_pred_df = evaluate_one_setting(
                target_tables=target_tables,
                candidate_tables=candidate_tables,
                y_target=y_target,
                edge_count=edge_count,
                cfg=cfg,
                feature_set_name=feature_set_name,
                run_dir=run_dir,
            )
            all_metrics.append(metrics)
            all_per_farm.append(per_farm_df)

            # 只保存 all 的 sample 预测，避免文件过多。
            if feature_set_name == "all":
                sample_pred_df.to_csv(
                    run_dir / "tables" / f"sample_predictions_edge_{edge_count}_all.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                sample_pred_df.to_csv(
                    run_dir / "vis_data" / f"sample_prediction_curve_edge_{edge_count}_all.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            print(f"     MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

    if not all_metrics:
        raise RuntimeError("没有任何 edge_count 被成功评估。请检查零件库目录和 split 设置。")

    summary_df = pd.DataFrame(all_metrics).sort_values(["feature_set", "edge_count"]).reset_index(drop=True)
    summary_df.to_csv(run_dir / "tables" / "piece_size_and_feature_ablation_summary.csv", index=False, encoding="utf-8-sig")

    per_farm_df = pd.concat(all_per_farm, axis=0, ignore_index=True)
    per_farm_df.to_csv(run_dir / "tables" / "per_farm_metrics.csv", index=False, encoding="utf-8-sig")

    critical = find_critical_point(summary_df, cfg)
    save_json(critical, run_dir / "metrics" / "critical_point.json")

    chosen_edge_count = int(critical["critical_point_by_elbow"]["edge_count"])
    importance_df = build_feature_importance(summary_df, chosen_edge_count)
    importance_df.to_csv(run_dir / "tables" / "feature_importance_at_critical_edge.csv", index=False, encoding="utf-8-sig")

    export_visualization_tables(summary_df, per_farm_df, importance_df, critical, run_dir)

    plot_size_sweep(summary_df, run_dir)
    plot_feature_importance(importance_df, run_dir)
    plot_ablation_heatmap(summary_df, run_dir)
    plot_source_ranking_from_details(run_dir, chosen_edge_count, cfg)

    print("\n===== 实验完成 =====")
    print(f"输出目录: {run_dir.resolve()}")
    print("\n最佳点/临界点:")
    print(json.dumps(critical, ensure_ascii=False, indent=2))
    print("\n特征重要性排序:")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    main()
