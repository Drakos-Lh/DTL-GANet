"""
Trend-Piece Matching Module after best piece-length selection
承接 piece_matching_full_sweep_E1_E6_save_visdata.py 的最佳零件长度学习结果，
基于 critical_point.json 自动选择最优 edge_count，并生成后续随机森林/解释分析可直接使用的匹配特征表。

核心定位：
1. 使用最佳 edge_count 对应的 low/mid 零件库；
2. 对目标 split 中每个风电场、每个趋势零件，去历史候选零件库中做 Top-k 相似匹配；
3. 输出 long 格式匹配明细：target_farm / target_sample_idx / target_piece_id / source_farm / source_sample_idx / source_piece_id / similarity / rank；
4. 输出 wide 格式随机森林特征表：一行对应一个目标零件，rank-1 到 rank-k 的 source 特征展开成列；
5. 默认 past_only=True，避免使用未来候选片段造成时间泄漏。

推荐运行：
python trend_piece_matching_from_best_edge.py \
  --critical-json outputs/piece_matching_full_sweep_E1_E6/metrics/critical_point.json \
  --data-dir vmd_data \
  --output-root outputs

如果 critical_point.json 暂时不在默认位置，也可以直接指定：
python trend_piece_matching_from_best_edge.py --edge-count 5 --data-dir vmd_data
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 1. 配置
# =========================================================
@dataclass
class Config:
    # 最佳零件长度学习器输出的 critical_point.json。
    critical_json: str = "outputs/piece_matching_full_sweep_E1_E6/metrics/critical_point.json"

    # 如果找不到 critical_json，则使用这个默认值。根据你当前实验，最佳/临界点均为 edge_count=5。
    default_edge_count: int = 5
    edge_count: Optional[int] = None

    # 零件库目录模板。edge_count=5 时默认读取 outputs/lowmid_piece_graphs_E5/{split}/farm_i_pieces.csv。
    piece_root_template: str = "outputs/lowmid_piece_graphs_E{edge_count}"
    fallback_piece_root_e3: str = "outputs/lowmid_piece_graphs"

    data_dir: str = "vmd_data"
    output_root: str = "outputs"
    run_name: str = "trend_piece_matching_best_edge"

    # 候选库默认只用 train，目标 split 可以同时生成 train / val / test 的特征。
    candidate_split: str = "train"
    target_splits: Tuple[str, ...] = ("train", "val", "test")

    num_farms: int = 6
    top_k: int = 5

    # 默认只做跨风电场匹配，对应“空间相关性”的显式学习。
    exclude_same_farm: bool = True

    # 默认只允许历史候选，避免未来信息泄漏。
    past_only: bool = True

    # 分块计算，避免相似度矩阵过大。
    target_block_size: int = 256
    candidate_block_size: int = 4096
    max_pieces_per_source: Optional[int] = None
    random_seed: int = 42

    # 相似度权重：沿用最佳零件长度学习器中的 all-feature 设置。
    weight_direction: float = 0.45
    weight_length_shape: float = 0.25
    weight_total_duration: float = 0.10
    weight_slope_shape: float = 0.20

    # 将 low/mid 下一步变化量转为 up / flat / down 标签时使用。
    # 数据是 scaled 域时，0.02 通常比 0 更稳，可按实验需要调整。
    flat_threshold: float = 0.02


DIR_MAP = {"up": 1.0, "down": -1.0, "flat": 0.0, "rise": 1.0, "fall": -1.0}
DIR_TEXT = {1: "up", 0: "flat", -1: "down"}


# =========================================================
# 2. 基础工具
# =========================================================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Trend-Piece Matching from best edge_count")
    parser.add_argument("--critical-json", type=str, default=Config.critical_json)
    parser.add_argument("--edge-count", type=int, default=None, help="手动指定 edge_count；优先级高于 critical_json")
    parser.add_argument("--default-edge-count", type=int, default=Config.default_edge_count)
    parser.add_argument("--piece-root-template", type=str, default=Config.piece_root_template)
    parser.add_argument("--fallback-piece-root-e3", type=str, default=Config.fallback_piece_root_e3)
    parser.add_argument("--data-dir", type=str, default=Config.data_dir)
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--candidate-split", type=str, default=Config.candidate_split)
    parser.add_argument("--target-splits", type=str, nargs="+", default=list(Config.target_splits))
    parser.add_argument("--num-farms", type=int, default=Config.num_farms)
    parser.add_argument("--top-k", type=int, default=Config.top_k)
    parser.add_argument("--allow-same-farm", action="store_true", help="允许同一风电场作为候选源；默认不允许")
    parser.add_argument("--no-past-only", action="store_true", help="允许使用非历史候选；默认 past_only=True")
    parser.add_argument("--target-block-size", type=int, default=Config.target_block_size)
    parser.add_argument("--candidate-block-size", type=int, default=Config.candidate_block_size)
    parser.add_argument("--max-pieces-per-source", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--weight-direction", type=float, default=Config.weight_direction)
    parser.add_argument("--weight-length-shape", type=float, default=Config.weight_length_shape)
    parser.add_argument("--weight-total-duration", type=float, default=Config.weight_total_duration)
    parser.add_argument("--weight-slope-shape", type=float, default=Config.weight_slope_shape)
    parser.add_argument("--flat-threshold", type=float, default=Config.flat_threshold)
    args = parser.parse_args()

    return Config(
        critical_json=args.critical_json,
        default_edge_count=args.default_edge_count,
        edge_count=args.edge_count,
        piece_root_template=args.piece_root_template,
        fallback_piece_root_e3=args.fallback_piece_root_e3,
        data_dir=args.data_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        candidate_split=args.candidate_split,
        target_splits=tuple(args.target_splits),
        num_farms=args.num_farms,
        top_k=args.top_k,
        exclude_same_farm=not args.allow_same_farm,
        past_only=not args.no_past_only,
        target_block_size=args.target_block_size,
        candidate_block_size=args.candidate_block_size,
        max_pieces_per_source=args.max_pieces_per_source,
        random_seed=args.random_seed,
        weight_direction=args.weight_direction,
        weight_length_shape=args.weight_length_shape,
        weight_total_duration=args.weight_total_duration,
        weight_slope_shape=args.weight_slope_shape,
        flat_threshold=args.flat_threshold,
    )


def ensure_dirs(run_dir: Path) -> None:
    for name in ["metrics", "tables", "rf_features", "match_details"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_edge_count(cfg: Config) -> Tuple[int, str]:
    if cfg.edge_count is not None:
        return int(cfg.edge_count), "manual_arg"

    p = Path(cfg.critical_json)
    if p.exists():
        data = load_json(p)
        if "critical_point_by_elbow" in data and "edge_count" in data["critical_point_by_elbow"]:
            return int(data["critical_point_by_elbow"]["edge_count"]), str(p)
        if "best_by_rmse" in data and "edge_count" in data["best_by_rmse"]:
            return int(data["best_by_rmse"]["edge_count"]), str(p)

    return int(cfg.default_edge_count), "fallback_default_edge_count"


def resolve_piece_root(cfg: Config, edge_count: int) -> Path:
    root = Path(cfg.piece_root_template.format(edge_count=edge_count))
    if root.exists():
        return root

    if edge_count == 3:
        fallback = Path(cfg.fallback_piece_root_e3)
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"未找到 edge_count={edge_count} 的零件库目录。应存在: {root}"
    )


def required_columns(edge_count: int) -> List[str]:
    cols = ["split", "farm_id", "sample_idx", "piece_local_id", "start_time_idx", "end_time_idx"]
    for e in range(edge_count):
        cols += [f"edge_{e}_direction", f"edge_{e}_length", f"edge_{e}_avg_slope"]
    return cols


def direction_to_code(x) -> float:
    if isinstance(x, str):
        key = x.strip().lower()
        if key in DIR_MAP:
            return float(DIR_MAP[key])
    try:
        v = float(x)
        if v > 0:
            return 1.0
        if v < 0:
            return -1.0
        return 0.0
    except Exception as exc:
        raise ValueError(f"无法解析方向值: {x}") from exc


def direction_label_from_delta(delta: float, threshold: float) -> Tuple[int, str]:
    if delta > threshold:
        return 1, "up"
    if delta < -threshold:
        return -1, "down"
    return 0, "flat"


# =========================================================
# 3. 数据读取与特征构建
# =========================================================
def load_y(data_dir: Path, split: str) -> Optional[np.ndarray]:
    path = data_dir / f"{split}_y_lowmid.npy"
    if not path.exists():
        print(f"[WARN] 未找到 {path}，将无法生成 y / direction 标签。")
        return None
    y = np.load(path)
    if y.ndim == 3:
        y = y[:, 0, :]
    if y.ndim != 2:
        raise ValueError(f"{path} 的 shape 应为 (N, 1, farms) 或 (N, farms)，实际为 {y.shape}")
    return y.astype(np.float32)


def load_current_lowmid_from_x(data_dir: Path, split: str) -> Optional[np.ndarray]:
    path = data_dir / f"{split}_X_lowmid.npy"
    if not path.exists():
        print(f"[WARN] 未找到 {path}，将无法生成 current_y / future_delta / direction 标签。")
        return None
    x = np.load(path)
    if x.ndim != 4:
        raise ValueError(f"{path} 的 shape 应为 (N, window, farms, 2)，实际为 {x.shape}")
    # low + mid 的当前窗口最后一步，仍在 scaled 域。
    cur = x[:, -1, :, :].sum(axis=-1)
    return cur.astype(np.float32)


def compute_split_offsets(data_dir: Path, splits: Sequence[str]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    cursor = 0
    for split in ["train", "val", "test"]:
        if split not in splits:
            # 即使本轮不生成该 split，也尽量读取长度，以便计算 val/test 的全局时间索引。
            pass
        y = load_y(data_dir, split)
        offsets[split] = cursor
        if y is not None:
            cursor += len(y)
    return offsets


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


def matrix_from_edges(df: pd.DataFrame, edge_count: int, field: str) -> np.ndarray:
    return np.stack(
        [df[f"edge_{e}_{field}"].to_numpy(dtype=np.float32) for e in range(edge_count)],
        axis=1,
    )


def direction_matrix(df: pd.DataFrame, edge_count: int) -> np.ndarray:
    return np.stack(
        [df[f"edge_{e}_direction"].map(direction_to_code).to_numpy(dtype=np.float32) for e in range(edge_count)],
        axis=1,
    )


def normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.sum(np.abs(arr), axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def safe_take_2d(arr: Optional[np.ndarray], sample_idx: int, farm_idx: int) -> float:
    if arr is None:
        return float("nan")
    if sample_idx < 0 or sample_idx >= len(arr) or farm_idx < 0 or farm_idx >= arr.shape[1]:
        return float("nan")
    return float(arr[sample_idx, farm_idx])


def piece_signature_text(row: pd.Series, edge_count: int, prefix: str = "") -> str:
    dirs = [str(row[f"{prefix}edge_{e}_direction"]) for e in range(edge_count)]
    lens = [int(row[f"{prefix}edge_{e}_length"]) for e in range(edge_count)]
    slopes = [float(row[f"{prefix}edge_{e}_avg_slope"]) for e in range(edge_count)]
    slope_text = ",".join(f"{v:.4f}" for v in slopes)
    return f"[{','.join(dirs)}] | len=({','.join(map(str, lens))}) | slope=({slope_text})"


def build_feature_bank(
    df: pd.DataFrame,
    split: str,
    edge_count: int,
    y_split: Optional[np.ndarray],
    current_split: Optional[np.ndarray],
    split_offset: int,
    cfg: Config,
) -> pd.DataFrame:
    bank = df.copy().reset_index(drop=True)
    bank["split"] = split
    bank["global_sample_idx"] = bank["sample_idx"].astype(int) + int(split_offset)

    dirs = direction_matrix(bank, edge_count)
    lens = matrix_from_edges(bank, edge_count, "length")
    slopes = matrix_from_edges(bank, edge_count, "avg_slope")
    lens_norm = normalize_rows(lens)
    slope_mag_norm = normalize_rows(np.abs(slopes))

    for e in range(edge_count):
        bank[f"dir_code_{e}"] = dirs[:, e]
        bank[f"len_norm_{e}"] = lens_norm[:, e]
        bank[f"slope_mag_norm_{e}"] = slope_mag_norm[:, e]

    bank["total_duration"] = lens.sum(axis=1)

    future_y, current_y, delta_y, dir_code, dir_label = [], [], [], [], []
    valid_rows = []

    for _, row in bank.iterrows():
        sample_idx = int(row["sample_idx"])
        farm_idx = int(row["farm_id"]) - 1
        fy = safe_take_2d(y_split, sample_idx, farm_idx)
        cy = safe_take_2d(current_split, sample_idx, farm_idx)
        dy = fy - cy if not (math.isnan(fy) or math.isnan(cy)) else float("nan")
        if math.isnan(dy):
            dc, dl = 999, "unknown"
        else:
            dc, dl = direction_label_from_delta(dy, cfg.flat_threshold)
        future_y.append(fy)
        current_y.append(cy)
        delta_y.append(dy)
        dir_code.append(dc)
        dir_label.append(dl)

        # 如果 y 缺失，不直接丢弃，因为仍然可以输出匹配关系；RF 标签列会是 NaN。
        valid_rows.append(True)

    bank["future_y_lowmid"] = future_y
    bank["current_y_lowmid"] = current_y
    bank["future_delta_lowmid"] = delta_y
    bank["future_direction_code"] = dir_code
    bank["future_direction_label"] = dir_label
    return bank[np.array(valid_rows, dtype=bool)].reset_index(drop=True)


def maybe_subsample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.max_pieces_per_source is None or len(df) <= cfg.max_pieces_per_source:
        return df.reset_index(drop=True)
    return df.sample(cfg.max_pieces_per_source, random_state=cfg.random_seed).reset_index(drop=True)


def feature_arrays(df: pd.DataFrame, edge_count: int) -> Dict[str, np.ndarray]:
    return {
        "dir": df[[f"dir_code_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "len": df[[f"len_norm_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "total": df[["total_duration"]].to_numpy(dtype=np.float32),
        "slope": df[[f"slope_mag_norm_{e}" for e in range(edge_count)]].to_numpy(dtype=np.float32),
        "global_sample_idx": df[["global_sample_idx"]].to_numpy(dtype=np.int64),
    }


# =========================================================
# 4. 相似度计算与 Top-k 匹配
# =========================================================
def normalized_weights(cfg: Config) -> Dict[str, float]:
    raw = {
        "direction": float(cfg.weight_direction),
        "length_shape": float(cfg.weight_length_shape),
        "total_duration": float(cfg.weight_total_duration),
        "slope_shape": float(cfg.weight_slope_shape),
    }
    s = sum(raw.values())
    if s <= 0:
        raise ValueError(f"相似度权重之和必须大于 0: {raw}")
    return {k: v / s for k, v in raw.items()}


def compute_similarity_components(
    target: Dict[str, np.ndarray],
    cand: Dict[str, np.ndarray],
    cfg: Config,
) -> Dict[str, np.ndarray]:
    direction_match = (target["dir"][:, None, :] == cand["dir"][None, :, :]).mean(axis=2, dtype=np.float32)

    len_l1 = np.abs(target["len"][:, None, :] - cand["len"][None, :, :]).mean(axis=2, dtype=np.float32)
    length_shape_sim = 1.0 - np.clip(len_l1, 0.0, 1.0)

    total_diff = np.abs(target["total"] - cand["total"].T)
    denom = np.maximum(np.maximum(target["total"], cand["total"].T), 1.0)
    total_duration_sim = 1.0 - np.clip(total_diff / denom, 0.0, 1.0)

    slope_l1 = np.abs(target["slope"][:, None, :] - cand["slope"][None, :, :]).mean(axis=2, dtype=np.float32)
    slope_shape_sim = 1.0 - np.clip(slope_l1, 0.0, 1.0)

    w = normalized_weights(cfg)
    total_score = (
        w["direction"] * direction_match
        + w["length_shape"] * length_shape_sim
        + w["total_duration"] * total_duration_sim
        + w["slope_shape"] * slope_shape_sim
    )

    return {
        "similarity": total_score.astype(np.float32),
        "direction_sim": direction_match.astype(np.float32),
        "length_shape_sim": length_shape_sim.astype(np.float32),
        "total_duration_sim": total_duration_sim.astype(np.float32),
        "slope_shape_sim": slope_shape_sim.astype(np.float32),
    }


def take_topk_including_components(
    best: Dict[str, np.ndarray],
    new: Dict[str, np.ndarray],
    new_indices: np.ndarray,
    k: int,
) -> Dict[str, np.ndarray]:
    # 合并旧 best 与当前 candidate block。
    merged_similarity = np.concatenate([best["similarity"], new["similarity"]], axis=1)
    merged_indices = np.concatenate([best["candidate_index"], new_indices], axis=1)

    comp_names = ["direction_sim", "length_shape_sim", "total_duration_sim", "slope_shape_sim"]
    merged_comps = {
        name: np.concatenate([best[name], new[name]], axis=1)
        for name in comp_names
    }

    valid_width = merged_similarity.shape[1]
    actual_k = min(k, valid_width)
    part = np.argpartition(-merged_similarity, kth=actual_k - 1, axis=1)[:, :actual_k]
    row_idx = np.arange(merged_similarity.shape[0])[:, None]

    out = {
        "similarity": merged_similarity[row_idx, part],
        "candidate_index": merged_indices[row_idx, part],
    }
    for name in comp_names:
        out[name] = merged_comps[name][row_idx, part]

    order = np.argsort(-out["similarity"], axis=1)
    out = {name: arr[row_idx, order] for name, arr in out.items()}
    return out


def streaming_topk_for_target_block(
    target_block_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    edge_count: int,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    target = feature_arrays(target_block_df, edge_count)
    n_t = len(target_block_df)
    k = min(cfg.top_k, len(cand_df))
    if k <= 0:
        raise ValueError("候选零件库为空，无法匹配。")

    best = {
        "similarity": np.full((n_t, k), -np.inf, dtype=np.float32),
        "candidate_index": np.full((n_t, k), -1, dtype=np.int32),
        "direction_sim": np.full((n_t, k), np.nan, dtype=np.float32),
        "length_shape_sim": np.full((n_t, k), np.nan, dtype=np.float32),
        "total_duration_sim": np.full((n_t, k), np.nan, dtype=np.float32),
        "slope_shape_sim": np.full((n_t, k), np.nan, dtype=np.float32),
    }

    target_global = target["global_sample_idx"].astype(np.int64)

    for c_start in range(0, len(cand_df), cfg.candidate_block_size):
        c_end = min(c_start + cfg.candidate_block_size, len(cand_df))
        cand_block_df = cand_df.iloc[c_start:c_end].reset_index(drop=True)
        cand = feature_arrays(cand_block_df, edge_count)
        comps = compute_similarity_components(target, cand, cfg)

        if cfg.past_only:
            cand_global = cand["global_sample_idx"].reshape(1, -1)
            valid_mask = cand_global < target_global
            for name in comps.keys():
                comps[name] = np.where(valid_mask, comps[name], -np.inf if name == "similarity" else np.nan)

        local_idx = np.arange(c_start, c_end, dtype=np.int32)[None, :].repeat(n_t, axis=0)
        best = take_topk_including_components(best, comps, local_idx, k)

    return best


# =========================================================
# 5. 输出 long 匹配表与 wide RF 特征表
# =========================================================
def add_edge_feature_columns(row_prefix: str, src_row: pd.Series, out: Dict, edge_count: int) -> None:
    for e in range(edge_count):
        out[f"{row_prefix}_edge_{e}_direction"] = src_row[f"edge_{e}_direction"]
        out[f"{row_prefix}_edge_{e}_dir_code"] = float(src_row[f"dir_code_{e}"])
        out[f"{row_prefix}_edge_{e}_length"] = float(src_row[f"edge_{e}_length"])
        out[f"{row_prefix}_edge_{e}_len_norm"] = float(src_row[f"len_norm_{e}"])
        out[f"{row_prefix}_edge_{e}_avg_slope"] = float(src_row[f"edge_{e}_avg_slope"])
        out[f"{row_prefix}_edge_{e}_slope_mag_norm"] = float(src_row[f"slope_mag_norm_{e}"])
    out[f"{row_prefix}_total_duration"] = float(src_row["total_duration"])


def build_matches_for_split(
    target_split: str,
    target_tables: Dict[int, pd.DataFrame],
    candidate_tables: Dict[int, pd.DataFrame],
    edge_count: int,
    cfg: Config,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for target_farm in range(1, cfg.num_farms + 1):
        target_df = target_tables[target_farm].reset_index(drop=True)
        candidate_frames = []
        for source_farm, df in candidate_tables.items():
            if cfg.exclude_same_farm and source_farm == target_farm:
                continue
            candidate_frames.append(maybe_subsample(df, cfg))
        if not candidate_frames:
            raise ValueError(f"target_farm={target_farm} 没有可用候选源。")
        cand_df = pd.concat(candidate_frames, axis=0, ignore_index=True)

        print(f"  target_split={target_split}, farm={target_farm}, target_pieces={len(target_df)}, candidates={len(cand_df)}")

        for t_start in range(0, len(target_df), cfg.target_block_size):
            t_end = min(t_start + cfg.target_block_size, len(target_df))
            target_block_df = target_df.iloc[t_start:t_end].reset_index(drop=True)
            best = streaming_topk_for_target_block(target_block_df, cand_df, edge_count, cfg)

            for i_local in range(len(target_block_df)):
                trow = target_block_df.iloc[i_local]
                for rank in range(best["similarity"].shape[1]):
                    sim = float(best["similarity"][i_local, rank])
                    cand_idx = int(best["candidate_index"][i_local, rank])
                    if cand_idx < 0 or not np.isfinite(sim):
                        continue
                    crow = cand_df.iloc[cand_idx]

                    out = {
                        "target_split": target_split,
                        "candidate_split": cfg.candidate_split,
                        "edge_count": edge_count,
                        "node_count": edge_count + 1,
                        "rank": rank + 1,

                        "target_farm": int(target_farm),
                        "target_sample_idx": int(trow["sample_idx"]),
                        "target_global_sample_idx": int(trow["global_sample_idx"]),
                        "target_piece_id": int(trow["piece_local_id"]),
                        "target_start_time_idx": int(trow["start_time_idx"]),
                        "target_end_time_idx": int(trow["end_time_idx"]),
                        "target_signature": piece_signature_text(trow, edge_count),
                        "target_current_y_lowmid": float(trow["current_y_lowmid"]),
                        "target_future_y_lowmid": float(trow["future_y_lowmid"]),
                        "target_future_delta_lowmid": float(trow["future_delta_lowmid"]),
                        "target_future_direction_code": int(trow["future_direction_code"]),
                        "target_future_direction_label": str(trow["future_direction_label"]),

                        "source_split": str(crow["split"]),
                        "source_farm": int(crow["farm_id"]),
                        "source_sample_idx": int(crow["sample_idx"]),
                        "source_global_sample_idx": int(crow["global_sample_idx"]),
                        "source_piece_id": int(crow["piece_local_id"]),
                        "source_start_time_idx": int(crow["start_time_idx"]),
                        "source_end_time_idx": int(crow["end_time_idx"]),
                        "source_signature": piece_signature_text(crow, edge_count),
                        "source_current_y_lowmid": float(crow["current_y_lowmid"]),
                        "source_future_y_lowmid": float(crow["future_y_lowmid"]),
                        "source_future_delta_lowmid": float(crow["future_delta_lowmid"]),
                        "source_future_direction_code": int(crow["future_direction_code"]),
                        "source_future_direction_label": str(crow["future_direction_label"]),

                        "lag_samples": int(trow["global_sample_idx"] - crow["global_sample_idx"]),
                        "similarity": sim,
                        "direction_sim": float(best["direction_sim"][i_local, rank]),
                        "length_shape_sim": float(best["length_shape_sim"][i_local, rank]),
                        "total_duration_sim": float(best["total_duration_sim"][i_local, rank]),
                        "slope_shape_sim": float(best["slope_shape_sim"][i_local, rank]),
                    }
                    add_edge_feature_columns("target", trow, out, edge_count)
                    add_edge_feature_columns("source", crow, out, edge_count)
                    rows.append(out)

    return pd.DataFrame(rows)


def build_rf_wide_features(matches_long: pd.DataFrame, edge_count: int, top_k: int) -> pd.DataFrame:
    if matches_long.empty:
        return pd.DataFrame()

    id_cols = [
        "target_split", "target_farm", "target_sample_idx", "target_global_sample_idx", "target_piece_id",
        "target_start_time_idx", "target_end_time_idx", "edge_count", "node_count",
    ]

    target_cols = [
        "target_signature",
        "target_current_y_lowmid", "target_future_y_lowmid", "target_future_delta_lowmid",
        "target_future_direction_code", "target_future_direction_label",
        "target_total_duration",
    ]
    for e in range(edge_count):
        target_cols += [
            f"target_edge_{e}_direction",
            f"target_edge_{e}_dir_code",
            f"target_edge_{e}_length",
            f"target_edge_{e}_len_norm",
            f"target_edge_{e}_avg_slope",
            f"target_edge_{e}_slope_mag_norm",
        ]

    rows: List[Dict] = []
    group_cols = id_cols
    for key, sub in matches_long.sort_values("rank").groupby(group_cols, dropna=False):
        first = sub.iloc[0]
        out = {col: first[col] for col in id_cols + target_cols if col in first.index}

        sims: List[float] = []
        source_direction_votes: List[int] = []
        source_delta_votes: List[float] = []

        for rank in range(1, top_k + 1):
            rsub = sub[sub["rank"] == rank]
            if rsub.empty:
                out[f"rank_{rank}_similarity"] = np.nan
                out[f"rank_{rank}_source_farm"] = np.nan
                out[f"rank_{rank}_source_sample_idx"] = np.nan
                out[f"rank_{rank}_source_piece_id"] = np.nan
                out[f"rank_{rank}_lag_samples"] = np.nan
                continue

            r = rsub.iloc[0]
            sim = float(r["similarity"])
            sims.append(sim)
            if int(r["source_future_direction_code"]) != 999:
                source_direction_votes.append(int(r["source_future_direction_code"]))
            if np.isfinite(float(r["source_future_delta_lowmid"])):
                source_delta_votes.append(float(r["source_future_delta_lowmid"]))

            out[f"rank_{rank}_similarity"] = sim
            out[f"rank_{rank}_direction_sim"] = float(r["direction_sim"])
            out[f"rank_{rank}_length_shape_sim"] = float(r["length_shape_sim"])
            out[f"rank_{rank}_total_duration_sim"] = float(r["total_duration_sim"])
            out[f"rank_{rank}_slope_shape_sim"] = float(r["slope_shape_sim"])
            out[f"rank_{rank}_source_split"] = r["source_split"]
            out[f"rank_{rank}_source_farm"] = int(r["source_farm"])
            out[f"rank_{rank}_source_sample_idx"] = int(r["source_sample_idx"])
            out[f"rank_{rank}_source_global_sample_idx"] = int(r["source_global_sample_idx"])
            out[f"rank_{rank}_source_piece_id"] = int(r["source_piece_id"])
            out[f"rank_{rank}_lag_samples"] = int(r["lag_samples"])
            out[f"rank_{rank}_source_future_y_lowmid"] = float(r["source_future_y_lowmid"])
            out[f"rank_{rank}_source_future_delta_lowmid"] = float(r["source_future_delta_lowmid"])
            out[f"rank_{rank}_source_future_direction_code"] = int(r["source_future_direction_code"])
            out[f"rank_{rank}_source_total_duration"] = float(r["source_total_duration"])

            for e in range(edge_count):
                out[f"rank_{rank}_source_edge_{e}_dir_code"] = float(r[f"source_edge_{e}_dir_code"])
                out[f"rank_{rank}_source_edge_{e}_length"] = float(r[f"source_edge_{e}_length"])
                out[f"rank_{rank}_source_edge_{e}_len_norm"] = float(r[f"source_edge_{e}_len_norm"])
                out[f"rank_{rank}_source_edge_{e}_avg_slope"] = float(r[f"source_edge_{e}_avg_slope"])
                out[f"rank_{rank}_source_edge_{e}_slope_mag_norm"] = float(r[f"source_edge_{e}_slope_mag_norm"])

        sim_arr = np.array(sims, dtype=np.float32)
        out["topk_similarity_mean"] = float(np.nanmean(sim_arr)) if len(sim_arr) else np.nan
        out["topk_similarity_max"] = float(np.nanmax(sim_arr)) if len(sim_arr) else np.nan
        out["topk_similarity_min"] = float(np.nanmin(sim_arr)) if len(sim_arr) else np.nan
        out["topk_similarity_std"] = float(np.nanstd(sim_arr)) if len(sim_arr) else np.nan
        out["top1_top2_similarity_gap"] = float(sim_arr[0] - sim_arr[1]) if len(sim_arr) >= 2 else np.nan
        out["matched_rank_count"] = int(len(sim_arr))

        # 相似度加权的历史后续变化，用于 RF 的强特征。
        if len(sim_arr) and source_delta_votes:
            delta_arr = np.array(source_delta_votes[: len(sim_arr)], dtype=np.float32)
            usable_sims = sim_arr[: len(delta_arr)]
            w = np.maximum(usable_sims, 1e-6)
            w = w / max(float(w.sum()), 1e-6)
            out["topk_weighted_source_future_delta"] = float((w * delta_arr).sum())
            out["topk_mean_source_future_delta"] = float(delta_arr.mean())
        else:
            out["topk_weighted_source_future_delta"] = np.nan
            out["topk_mean_source_future_delta"] = np.nan

        if source_direction_votes:
            votes = np.array(source_direction_votes, dtype=np.int32)
            out["topk_source_up_ratio"] = float(np.mean(votes == 1))
            out["topk_source_flat_ratio"] = float(np.mean(votes == 0))
            out["topk_source_down_ratio"] = float(np.mean(votes == -1))
        else:
            out["topk_source_up_ratio"] = np.nan
            out["topk_source_flat_ratio"] = np.nan
            out["topk_source_down_ratio"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def build_source_summary(matches_long: pd.DataFrame) -> pd.DataFrame:
    if matches_long.empty:
        return pd.DataFrame()
    out = (
        matches_long
        .groupby(["target_split", "target_farm", "source_farm"], as_index=False)
        .agg(
            matched_count=("similarity", "count"),
            avg_similarity=("similarity", "mean"),
            max_similarity=("similarity", "max"),
            avg_lag_samples=("lag_samples", "mean"),
        )
        .sort_values(["target_split", "target_farm", "avg_similarity", "matched_count"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )
    out["source_rank_within_target"] = out.groupby(["target_split", "target_farm"]).cumcount() + 1
    return out


def build_lag_summary(matches_long: pd.DataFrame) -> pd.DataFrame:
    if matches_long.empty:
        return pd.DataFrame()
    return (
        matches_long
        .groupby(["target_split", "target_farm", "source_farm", "lag_samples"], as_index=False)
        .agg(lag_count=("similarity", "count"), avg_similarity=("similarity", "mean"))
        .sort_values(["target_split", "target_farm", "source_farm", "lag_samples"])
        .reset_index(drop=True)
    )


# =========================================================
# 6. 主流程
# =========================================================
def main() -> None:
    cfg = parse_args()
    edge_count, edge_source = select_edge_count(cfg)
    piece_root = resolve_piece_root(cfg, edge_count)

    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)

    data_dir = Path(cfg.data_dir)
    all_splits = sorted(set([cfg.candidate_split, *cfg.target_splits, "train", "val", "test"]), key=["train", "val", "test"].index)
    split_offsets = compute_split_offsets(data_dir, all_splits)

    config_dump = asdict(cfg)
    config_dump["selected_edge_count"] = edge_count
    config_dump["selected_node_count"] = edge_count + 1
    config_dump["edge_count_source"] = edge_source
    config_dump["piece_root"] = str(piece_root)
    config_dump["split_offsets"] = split_offsets
    save_json(config_dump, run_dir / "metrics" / "config.json")

    print("===== Trend-Piece Matching from Best Edge =====")
    print(f"selected edge_count={edge_count}, node_count={edge_count + 1}, source={edge_source}")
    print(f"piece_root={piece_root}")
    print(f"candidate_split={cfg.candidate_split}, target_splits={cfg.target_splits}")

    # 读取候选库。
    y_candidate = load_y(data_dir, cfg.candidate_split)
    cur_candidate = load_current_lowmid_from_x(data_dir, cfg.candidate_split)
    raw_candidate = load_piece_tables(piece_root, cfg.candidate_split, cfg.num_farms, edge_count)
    candidate_tables = {
        fid: build_feature_bank(
            df, cfg.candidate_split, edge_count, y_candidate, cur_candidate,
            split_offsets.get(cfg.candidate_split, 0), cfg
        )
        for fid, df in raw_candidate.items()
    }

    piece_counts = {
        "candidate": {f"farm_{fid}": int(len(df)) for fid, df in candidate_tables.items()},
        "targets": {},
    }

    all_long: List[pd.DataFrame] = []
    all_wide: List[pd.DataFrame] = []

    for target_split in cfg.target_splits:
        print(f"\n===== 生成 target_split={target_split} 的匹配特征 =====")
        y_target = load_y(data_dir, target_split)
        cur_target = load_current_lowmid_from_x(data_dir, target_split)
        raw_target = load_piece_tables(piece_root, target_split, cfg.num_farms, edge_count)
        target_tables = {
            fid: build_feature_bank(
                df, target_split, edge_count, y_target, cur_target,
                split_offsets.get(target_split, 0), cfg
            )
            for fid, df in raw_target.items()
        }
        piece_counts["targets"][target_split] = {f"farm_{fid}": int(len(df)) for fid, df in target_tables.items()}

        matches_long = build_matches_for_split(target_split, target_tables, candidate_tables, edge_count, cfg)
        matches_long_path = run_dir / "match_details" / f"{target_split}_matches_long.csv"
        matches_long.to_csv(matches_long_path, index=False, encoding="utf-8-sig")

        rf_wide = build_rf_wide_features(matches_long, edge_count, cfg.top_k)
        rf_wide_path = run_dir / "rf_features" / f"{target_split}_rf_features_wide.csv"
        rf_wide.to_csv(rf_wide_path, index=False, encoding="utf-8-sig")

        source_summary = build_source_summary(matches_long)
        source_summary.to_csv(run_dir / "tables" / f"{target_split}_source_farm_summary.csv", index=False, encoding="utf-8-sig")

        lag_summary = build_lag_summary(matches_long)
        lag_summary.to_csv(run_dir / "tables" / f"{target_split}_lag_distribution.csv", index=False, encoding="utf-8-sig")

        all_long.append(matches_long)
        all_wide.append(rf_wide)
        print(f"[OK] {target_split}: long={len(matches_long)}, wide={len(rf_wide)}")
        print(f"     long -> {matches_long_path}")
        print(f"     wide -> {rf_wide_path}")

    save_json(piece_counts, run_dir / "metrics" / "piece_counts.json")

    if all_long:
        all_long_df = pd.concat(all_long, axis=0, ignore_index=True)
        all_long_df.to_csv(run_dir / "match_details" / "all_splits_matches_long.csv", index=False, encoding="utf-8-sig")
        build_source_summary(all_long_df).to_csv(run_dir / "tables" / "all_splits_source_farm_summary.csv", index=False, encoding="utf-8-sig")
        build_lag_summary(all_long_df).to_csv(run_dir / "tables" / "all_splits_lag_distribution.csv", index=False, encoding="utf-8-sig")
    else:
        all_long_df = pd.DataFrame()

    if all_wide:
        all_wide_df = pd.concat(all_wide, axis=0, ignore_index=True)
        all_wide_df.to_csv(run_dir / "rf_features" / "all_splits_rf_features_wide.csv", index=False, encoding="utf-8-sig")
    else:
        all_wide_df = pd.DataFrame()

    overview = {
        "task": "根据最佳 edge_count 生成趋势零件 Top-k 匹配特征",
        "selected_edge_count": edge_count,
        "selected_node_count": edge_count + 1,
        "edge_count_source": edge_source,
        "piece_root": str(piece_root),
        "candidate_split": cfg.candidate_split,
        "target_splits": list(cfg.target_splits),
        "top_k": cfg.top_k,
        "exclude_same_farm": cfg.exclude_same_farm,
        "past_only": cfg.past_only,
        "flat_threshold": cfg.flat_threshold,
        "long_match_rows": int(len(all_long_df)),
        "wide_feature_rows": int(len(all_wide_df)),
        "output_files": {
            "all_matches_long": str(run_dir / "match_details" / "all_splits_matches_long.csv"),
            "all_rf_features_wide": str(run_dir / "rf_features" / "all_splits_rf_features_wide.csv"),
            "source_farm_summary": str(run_dir / "tables" / "all_splits_source_farm_summary.csv"),
            "lag_distribution": str(run_dir / "tables" / "all_splits_lag_distribution.csv"),
        },
    }
    save_json(overview, run_dir / "metrics" / "overview.json")

    print("\n===== 完成 =====")
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    print(f"输出目录: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
