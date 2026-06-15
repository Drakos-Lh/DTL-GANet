"""
Build GAT-ready star-graph datasets from trend-piece Top-k matching features.

定位：
- 作为 2_trend_piece_matching_from_best_edge.py 之后的“图结构学习接口层”；
- 不改动 1 号脚本，也不破坏 RF baseline 的 wide 特征表；
- 读取 *_rf_features_wide.csv，将每个目标趋势零件构造成一个星型图：
    target 趋势零件节点 + Top-k 历史相似趋势零件邻居节点；
- 输出 train/val/test 的 .npz 图数据，供 3_gat_lowmid_regressor_from_piece_graphs.py 训练。

趋势零件说明：
这里的“趋势零件”不是机械零件，而是风电功率时间序列中的局部时序片段。
每个片段由若干条趋势边组成，包含方向、长度、斜率、持续时间等形态特征。

推荐运行：
python 2b_build_gat_graph_dataset_from_piece_features.py \
  --feature-dir outputs/trend_piece_matching_best_edge/rf_features \
  --output-root outputs

输入文件默认：
outputs/trend_piece_matching_best_edge/rf_features/train_rf_features_wide.csv
outputs/trend_piece_matching_best_edge/rf_features/val_rf_features_wide.csv
outputs/trend_piece_matching_best_edge/rf_features/test_rf_features_wide.csv

输出目录默认：
outputs/gat_piece_graph_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    feature_dir: str = "outputs/trend_piece_matching_best_edge/rf_features"
    output_root: str = "outputs"
    run_name: str = "gat_piece_graph_dataset"

    train_file: str = "train_rf_features_wide.csv"
    val_file: str = "val_rf_features_wide.csv"
    test_file: str = "test_rf_features_wide.csv"

    num_farms: int = 6

    # 标签：建议用 delta，预测下一步 low/mid 变化量，再加回 current 得到 future_y。
    prediction_target: str = "delta"  # delta / future_y

    # 缺失邻居处理：不足 Top-k 时保留 mask=0 的空邻居。
    keep_incomplete_topk: bool = True

    # 是否在源节点特征中保留 source 历史后续演化信息。
    # 这是 RF baseline 的关键逻辑：历史相似片段后续怎么走，可为当前趋势提供参考。
    include_source_future_info: bool = True

    # 是否把 source_sample_idx / lag_samples 显式作为边特征。
    include_time_index_features: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build GAT graph dataset from trend-piece matching wide features")
    parser.add_argument("--feature-dir", type=str, default=Config.feature_dir)
    parser.add_argument("--output-root", type=str, default=Config.output_root)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--train-file", type=str, default=Config.train_file)
    parser.add_argument("--val-file", type=str, default=Config.val_file)
    parser.add_argument("--test-file", type=str, default=Config.test_file)
    parser.add_argument("--num-farms", type=int, default=Config.num_farms)
    parser.add_argument("--prediction-target", choices=["delta", "future_y"], default=Config.prediction_target)
    parser.add_argument("--drop-incomplete-topk", action="store_true")
    parser.add_argument("--exclude-source-future-info", action="store_true")
    parser.add_argument("--exclude-time-index-features", action="store_true")
    args = parser.parse_args()

    return Config(
        feature_dir=args.feature_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        num_farms=args.num_farms,
        prediction_target=args.prediction_target,
        keep_incomplete_topk=not args.drop_incomplete_topk,
        include_source_future_info=not args.exclude_source_future_info,
        include_time_index_features=not args.exclude_time_index_features,
    )


def ensure_dirs(run_dir: Path) -> None:
    for sub in ["graphs", "metrics", "tables"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到输入特征表: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"输入特征表为空: {path}")
    return df


def to_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return default
    val = row[col]
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def to_int(row: pd.Series, col: str, default: int = 0) -> int:
    if col not in row.index:
        return default
    try:
        if pd.isna(row[col]):
            return default
        return int(float(row[col]))
    except Exception:
        return default


def infer_edge_count(df: pd.DataFrame) -> int:
    edges = []
    for col in df.columns:
        m = re.match(r"target_edge_(\d+)_dir_code$", col)
        if m:
            edges.append(int(m.group(1)))
    if not edges:
        # 兜底：从 edge_count 列读取
        if "edge_count" in df.columns:
            return int(df["edge_count"].dropna().iloc[0])
        raise ValueError("无法从 wide 特征表推断 edge_count。")
    return max(edges) + 1


def infer_top_k(df: pd.DataFrame) -> int:
    ranks = []
    for col in df.columns:
        m = re.match(r"rank_(\d+)_similarity$", col)
        if m:
            ranks.append(int(m.group(1)))
    if not ranks:
        raise ValueError("无法从 wide 特征表推断 Top-k。未找到 rank_*_similarity 列。")
    return max(ranks)


def basic_required_columns() -> List[str]:
    return [
        "target_farm",
        "target_sample_idx",
        "target_current_y_lowmid",
        "target_future_y_lowmid",
        "target_future_delta_lowmid",
    ]


def validate_df(df: pd.DataFrame, split: str) -> None:
    missing = [c for c in basic_required_columns() if c not in df.columns]
    if missing:
        raise KeyError(f"{split} 特征表缺少必要列: {missing}")


def build_target_node(row: pd.Series, edge_count: int, cfg: Config) -> List[float]:
    feat: List[float] = []
    for e in range(edge_count):
        feat.extend([
            to_float(row, f"target_edge_{e}_dir_code"),
            to_float(row, f"target_edge_{e}_length"),
            to_float(row, f"target_edge_{e}_len_norm"),
            to_float(row, f"target_edge_{e}_avg_slope"),
            to_float(row, f"target_edge_{e}_slope_mag_norm"),
        ])

    # 统一节点尾部语义：
    # total_duration, value_hint, farm_norm, rank_norm, similarity_hint, node_type
    # target 的 value_hint 用 current_y，source 的 value_hint 用历史后续 delta 或 future_y。
    farm_norm = (to_float(row, "target_farm") - 1.0) / max(cfg.num_farms - 1, 1)
    feat.extend([
        to_float(row, "target_total_duration"),
        to_float(row, "target_current_y_lowmid"),
        farm_norm,
        0.0,
        1.0,
        1.0,  # target node flag
    ])
    return feat


def build_source_node(row: pd.Series, rank: int, edge_count: int, cfg: Config) -> Tuple[List[float], bool]:
    sim = to_float(row, f"rank_{rank}_similarity", default=np.nan)
    source_farm = to_float(row, f"rank_{rank}_source_farm", default=np.nan)
    valid = math.isfinite(sim) and math.isfinite(source_farm) and sim > -1e20
    if not valid:
        # 返回定长空特征
        dummy_len = edge_count * 5 + 6
        return [0.0] * dummy_len, False

    feat: List[float] = []
    for e in range(edge_count):
        feat.extend([
            to_float(row, f"rank_{rank}_source_edge_{e}_dir_code"),
            to_float(row, f"rank_{rank}_source_edge_{e}_length"),
            to_float(row, f"rank_{rank}_source_edge_{e}_len_norm"),
            to_float(row, f"rank_{rank}_source_edge_{e}_avg_slope"),
            to_float(row, f"rank_{rank}_source_edge_{e}_slope_mag_norm"),
        ])

    if cfg.include_source_future_info:
        value_hint = to_float(row, f"rank_{rank}_source_future_delta_lowmid")
    else:
        value_hint = 0.0

    farm_norm = (source_farm - 1.0) / max(cfg.num_farms - 1, 1)
    rank_norm = (rank - 1.0) / max(infer_top_k_cached(row), 1)

    feat.extend([
        to_float(row, f"rank_{rank}_source_total_duration"),
        value_hint,
        farm_norm,
        rank_norm,
        sim,
        0.0,  # source node flag
    ])
    return feat, True


def infer_top_k_cached(row: pd.Series) -> int:
    # row 上没有 df 上下文，只用于 rank_norm 粗略归一化。
    ranks = []
    for col in row.index:
        m = re.match(r"rank_(\d+)_similarity$", col)
        if m:
            ranks.append(int(m.group(1)))
    return max(ranks) if ranks else 1


def build_edge_attr(row: pd.Series, rank: int, cfg: Config) -> List[float]:
    sim = to_float(row, f"rank_{rank}_similarity")
    source_farm = to_float(row, f"rank_{rank}_source_farm")
    farm_norm = (source_farm - 1.0) / max(cfg.num_farms - 1, 1) if math.isfinite(source_farm) else 0.0
    rank_norm = (rank - 1.0) / max(infer_top_k_cached(row), 1)

    lag = to_float(row, f"rank_{rank}_lag_samples")
    lag_log = math.copysign(math.log1p(abs(lag)), lag) if cfg.include_time_index_features else 0.0

    source_sample = to_float(row, f"rank_{rank}_source_sample_idx")
    source_sample_log = math.log1p(max(source_sample, 0.0)) if cfg.include_time_index_features else 0.0

    return [
        sim,
        to_float(row, f"rank_{rank}_direction_sim"),
        to_float(row, f"rank_{rank}_length_shape_sim"),
        to_float(row, f"rank_{rank}_total_duration_sim"),
        to_float(row, f"rank_{rank}_slope_shape_sim"),
        lag_log,
        source_sample_log,
        farm_norm,
        rank_norm,
    ]


def build_graph_arrays_for_split(df: pd.DataFrame, split: str, edge_count: int, top_k: int, cfg: Config) -> Dict[str, np.ndarray]:
    validate_df(df, split)

    node_x_rows: List[np.ndarray] = []
    edge_attr_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []

    y_delta: List[float] = []
    y_future: List[float] = []
    current_y: List[float] = []
    target_farm: List[int] = []
    target_sample_idx: List[int] = []
    target_piece_id: List[int] = []

    dropped = 0

    for _, row in df.iterrows():
        cy = to_float(row, "target_current_y_lowmid", default=np.nan)
        fy = to_float(row, "target_future_y_lowmid", default=np.nan)
        dy = to_float(row, "target_future_delta_lowmid", default=np.nan)
        if not (math.isfinite(cy) and math.isfinite(fy) and math.isfinite(dy)):
            dropped += 1
            continue

        target_node = build_target_node(row, edge_count, cfg)
        nodes = [target_node]
        edges = []
        masks = []

        for rank in range(1, top_k + 1):
            source_node, valid = build_source_node(row, rank, edge_count, cfg)
            nodes.append(source_node)
            edges.append(build_edge_attr(row, rank, cfg) if valid else [0.0] * 9)
            masks.append(1.0 if valid else 0.0)

        if (not cfg.keep_incomplete_topk) and sum(masks) < top_k:
            dropped += 1
            continue
        if sum(masks) <= 0:
            dropped += 1
            continue

        node_x_rows.append(np.asarray(nodes, dtype=np.float32))
        edge_attr_rows.append(np.asarray(edges, dtype=np.float32))
        mask_rows.append(np.asarray(masks, dtype=np.float32))

        y_delta.append(dy)
        y_future.append(fy)
        current_y.append(cy)
        target_farm.append(to_int(row, "target_farm"))
        target_sample_idx.append(to_int(row, "target_sample_idx"))
        target_piece_id.append(to_int(row, "target_piece_id", to_int(row, "target_piece_local_id", 0)))

    if not node_x_rows:
        raise ValueError(f"{split} 没有可用图样本，dropped={dropped}")

    out = {
        "node_x": np.stack(node_x_rows, axis=0).astype(np.float32),
        "edge_attr": np.stack(edge_attr_rows, axis=0).astype(np.float32),
        "neighbor_mask": np.stack(mask_rows, axis=0).astype(np.float32),
        "y_delta": np.asarray(y_delta, dtype=np.float32),
        "y_future": np.asarray(y_future, dtype=np.float32),
        "current_y": np.asarray(current_y, dtype=np.float32),
        "target_farm": np.asarray(target_farm, dtype=np.int64),
        "target_sample_idx": np.asarray(target_sample_idx, dtype=np.int64),
        "target_piece_id": np.asarray(target_piece_id, dtype=np.int64),
        "dropped_rows": np.asarray([dropped], dtype=np.int64),
    }

    if cfg.prediction_target == "delta":
        out["y"] = out["y_delta"].copy()
    else:
        out["y"] = out["y_future"].copy()
    return out


def standardize_graphs(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    train = splits["train"]

    node_mean = train["node_x"].reshape(-1, train["node_x"].shape[-1]).mean(axis=0)
    node_std = train["node_x"].reshape(-1, train["node_x"].shape[-1]).std(axis=0)
    node_std = np.where(node_std < 1e-6, 1.0, node_std)

    edge_mean = train["edge_attr"].reshape(-1, train["edge_attr"].shape[-1]).mean(axis=0)
    edge_std = train["edge_attr"].reshape(-1, train["edge_attr"].shape[-1]).std(axis=0)
    edge_std = np.where(edge_std < 1e-6, 1.0, edge_std)

    for data in splits.values():
        data["node_x_raw"] = data["node_x"].copy()
        data["edge_attr_raw"] = data["edge_attr"].copy()
        data["node_x"] = ((data["node_x"] - node_mean.reshape(1, 1, -1)) / node_std.reshape(1, 1, -1)).astype(np.float32)
        data["edge_attr"] = ((data["edge_attr"] - edge_mean.reshape(1, 1, -1)) / edge_std.reshape(1, 1, -1)).astype(np.float32)

    return {
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
    }


def save_npz(data: Dict[str, np.ndarray], path: Path) -> None:
    np.savez_compressed(path, **data)


def main() -> None:
    cfg = parse_args()
    feature_dir = Path(cfg.feature_dir)
    run_dir = Path(cfg.output_root) / cfg.run_name
    ensure_dirs(run_dir)

    files = {
        "train": feature_dir / cfg.train_file,
        "val": feature_dir / cfg.val_file,
        "test": feature_dir / cfg.test_file,
    }
    dfs = {split: read_csv(path) for split, path in files.items()}

    edge_count = infer_edge_count(dfs["train"])
    top_k = infer_top_k(dfs["train"])

    print("===== Build GAT graph dataset from trend-piece matching features =====")
    print(f"feature_dir={feature_dir}")
    print(f"edge_count={edge_count}, top_k={top_k}")

    splits = {
        split: build_graph_arrays_for_split(df, split, edge_count, top_k, cfg)
        for split, df in dfs.items()
    }
    norm_stats = standardize_graphs(splits)

    overview = {
        "task": "build GAT-ready star graph dataset from RF wide features",
        "feature_dir": str(feature_dir),
        "run_dir": str(run_dir),
        "edge_count": edge_count,
        "node_count_per_graph": top_k + 1,
        "top_k": top_k,
        "node_feature_dim": int(splits["train"]["node_x"].shape[-1]),
        "edge_feature_dim": int(splits["train"]["edge_attr"].shape[-1]),
        "prediction_target": cfg.prediction_target,
        "split_rows": {},
        "output_files": {},
    }

    for split, data in splits.items():
        out_path = run_dir / "graphs" / f"{split}_gat_graphs.npz"
        save_npz(data, out_path)
        overview["split_rows"][split] = {
            "graphs": int(data["node_x"].shape[0]),
            "dropped_rows": int(data["dropped_rows"][0]),
        }
        overview["output_files"][split] = str(out_path)

        # 元数据 CSV，便于调试与论文可视化对齐。
        meta = pd.DataFrame({
            "target_farm": data["target_farm"],
            "target_sample_idx": data["target_sample_idx"],
            "target_piece_id": data["target_piece_id"],
            "current_y": data["current_y"],
            "y_future": data["y_future"],
            "y_delta": data["y_delta"],
            "valid_neighbor_count": data["neighbor_mask"].sum(axis=1),
        })
        meta.to_csv(run_dir / "tables" / f"{split}_graph_meta.csv", index=False, encoding="utf-8-sig")

    save_json(asdict(cfg), run_dir / "metrics" / "config.json")
    save_json(norm_stats, run_dir / "metrics" / "normalization_stats.json")
    save_json(overview, run_dir / "metrics" / "overview.json")

    print("\n===== 完成 =====")
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
