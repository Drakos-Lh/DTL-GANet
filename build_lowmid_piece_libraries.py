"""
生成 low/mid 趋势零件库脚本

作用：
1. 从 vmd_data/{split}_X_lowmid.npy 读取 low/mid 序列；
2. 针对 edge_count = 1~6 生成不同尺寸的趋势零件库；
3. 输出到 outputs/lowmid_piece_graphs_E{edge_count}/{split}/farm_{i}_pieces.csv；
4. 输出列名与 piece_matching_upgrade_experiment.py 完全对齐。

推荐先直接运行：
python build_lowmid_piece_libraries.py

随后再运行：
python piece_matching_upgrade_experiment.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 1. 配置区：优先只改这里
# =========================================================
@dataclass
class Config:
    data_dir: str = "vmd_data"
    output_root: str = "outputs"
    output_template: str = "lowmid_piece_graphs_E{edge_count}"

    # 需要生成的零件尺寸：E3 = 3 条边 = 4 个节点
    edge_counts: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    splits: Tuple[str, ...] = ("train", "val", "test")

    # 数据形状通常为 (samples, 96, 6, 2)
    num_farms: int = 6

    # lowmid 序列如何合成：
    # "sum" = low + mid，推荐；"low" = 只用低频；"mid" = 只用中频
    series_mode: str = "sum"

    # 趋势边提取参数
    # flat_threshold = max(min_flat_eps, std(diff) * flat_ratio)
    # flat_ratio 越大，越容易把小波动视为 flat，零件更少、更稳定
    flat_ratio: float = 0.02
    min_flat_eps: float = 1e-6
    min_edge_len: int = 2

    # 每个样本窗口最多保留多少个零件。
    # 为了预测下一步，默认保留靠近窗口末端的最近零件，避免零件库爆炸。
    # 如果想保留全部，改为 None，但后续匹配会显著变慢。
    max_pieces_per_sample: int | None = 3
    keep_policy: str = "tail"  # tail / uniform / head / strongest

    # 降采样样本，控制零件库规模。
    # 严格实验可全部设为 1；电脑吃紧时 train 可设为 2 或 4。
    train_sample_stride: int = 2
    val_sample_stride: int = 1
    test_sample_stride: int = 1

    # 是否覆盖已有 CSV
    overwrite: bool = True


CFG = Config()


# =========================================================
# 2. 基础工具
# =========================================================
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_x_lowmid(data_dir: Path, split: str) -> np.ndarray:
    path = data_dir / f"{split}_X_lowmid.npy"
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件: {path}")
    x = np.load(path)
    if x.ndim != 4:
        raise ValueError(f"{path} 应为 4 维数组，例如 (N, 96, 6, 2)，实际 shape={x.shape}")
    return x.astype(np.float32)


def compose_lowmid_series(x_one: np.ndarray, mode: str) -> np.ndarray:
    """
    x_one: (T, C)，通常 C=2，分别是 low 和 mid。
    return: (T,)
    """
    if x_one.ndim != 2:
        raise ValueError(f"单个风电场输入应为 (T, C)，实际 shape={x_one.shape}")

    if mode == "sum":
        if x_one.shape[1] == 1:
            return x_one[:, 0]
        return x_one[:, 0] + x_one[:, 1]
    if mode == "low":
        return x_one[:, 0]
    if mode == "mid":
        if x_one.shape[1] < 2:
            raise ValueError("series_mode='mid' 需要 X_lowmid 最后一维至少为 2")
        return x_one[:, 1]

    raise ValueError(f"未知 series_mode={mode}，应为 sum / low / mid")


def direction_from_slope(slope: float, eps: float) -> str:
    if slope > eps:
        return "up"
    if slope < -eps:
        return "down"
    return "flat"


# =========================================================
# 3. 趋势边与趋势零件生成
# =========================================================
def extract_edges_from_series(series: np.ndarray, cfg: Config) -> List[Dict]:
    """
    将一个长度 T 的序列压缩为若干趋势边。
    每条边包含 start, end, direction, length, avg_slope。
    """
    y = np.asarray(series, dtype=np.float32).reshape(-1)
    T = len(y)
    if T < 2:
        return []

    diff = np.diff(y)
    eps = max(float(cfg.min_flat_eps), float(np.std(diff) * cfg.flat_ratio))

    signs = np.zeros_like(diff, dtype=np.int8)
    signs[diff > eps] = 1
    signs[diff < -eps] = -1

    # 连续相同 sign 合并成一条边
    raw_edges = []
    start = 0
    cur = int(signs[0])
    for i in range(1, len(signs)):
        s = int(signs[i])
        if s != cur:
            end = i
            raw_edges.append((start, end, cur))  # 覆盖 y[start] -> y[end]
            start = i
            cur = s
    raw_edges.append((start, T - 1, cur))

    # 先转成标准边
    edges = []
    for start, end, _ in raw_edges:
        if end <= start:
            continue
        length = int(end - start)
        slope = float((y[end] - y[start]) / max(length, 1))
        direction = direction_from_slope(slope, eps)
        edges.append({
            "start": int(start),
            "end": int(end),
            "direction": direction,
            "length": int(length),
            "avg_slope": float(slope),
        })

    # 过滤过短边：简单合并到前一条边，减少噪声造成的碎片
    merged: List[Dict] = []
    for edge in edges:
        if merged and edge["length"] < cfg.min_edge_len:
            prev = merged[-1]
            new_start = prev["start"]
            new_end = edge["end"]
            length = int(new_end - new_start)
            slope = float((y[new_end] - y[new_start]) / max(length, 1))
            merged[-1] = {
                "start": int(new_start),
                "end": int(new_end),
                "direction": direction_from_slope(slope, eps),
                "length": int(length),
                "avg_slope": float(slope),
            }
        else:
            merged.append(edge)

    return merged


def fallback_uniform_edges(series: np.ndarray, edge_count: int, cfg: Config) -> List[Dict]:
    """
    当转折点不足时，使用均匀切分兜底，保证每个样本至少能生成一个零件。
    """
    y = np.asarray(series, dtype=np.float32).reshape(-1)
    T = len(y)
    points = np.linspace(0, T - 1, edge_count + 1).round().astype(int)
    points = np.maximum.accumulate(points)

    diff = np.diff(y)
    eps = max(float(cfg.min_flat_eps), float(np.std(diff) * cfg.flat_ratio)) if len(diff) else cfg.min_flat_eps

    edges = []
    for e in range(edge_count):
        start = int(points[e])
        end = int(points[e + 1])
        if end <= start:
            end = min(start + 1, T - 1)
        length = int(max(end - start, 1))
        slope = float((y[end] - y[start]) / length)
        edges.append({
            "start": start,
            "end": end,
            "direction": direction_from_slope(slope, eps),
            "length": length,
            "avg_slope": slope,
        })
    return edges


def select_piece_windows(num_windows: int, max_keep: int | None, policy: str, edges: List[Dict], edge_count: int) -> List[int]:
    """
    返回保留哪些 piece 起点。一个起点 s 表示 edges[s:s+edge_count] 构成一个零件。
    """
    starts = list(range(num_windows))
    if max_keep is None or len(starts) <= max_keep:
        return starts

    if policy == "tail":
        return starts[-max_keep:]
    if policy == "head":
        return starts[:max_keep]
    if policy == "uniform":
        idx = np.linspace(0, len(starts) - 1, max_keep).round().astype(int)
        return [starts[i] for i in idx]
    if policy == "strongest":
        # 保留总变化幅度最大的若干零件
        scored = []
        for s in starts:
            piece_edges = edges[s:s + edge_count]
            score = sum(abs(float(ed["avg_slope"])) * int(ed["length"]) for ed in piece_edges)
            scored.append((score, s))
        scored.sort(reverse=True)
        chosen = sorted([s for _, s in scored[:max_keep]])
        return chosen

    raise ValueError(f"未知 keep_policy={policy}")


def build_pieces_for_sample(
    series: np.ndarray,
    split: str,
    farm_id: int,
    sample_idx: int,
    edge_count: int,
    cfg: Config,
) -> List[Dict]:
    edges = extract_edges_from_series(series, cfg)
    if len(edges) < edge_count:
        edges = fallback_uniform_edges(series, edge_count, cfg)

    num_windows = max(0, len(edges) - edge_count + 1)
    if num_windows <= 0:
        return []

    starts = select_piece_windows(
        num_windows=num_windows,
        max_keep=cfg.max_pieces_per_sample,
        policy=cfg.keep_policy,
        edges=edges,
        edge_count=edge_count,
    )

    rows = []
    local_id = 0
    for s in starts:
        piece_edges = edges[s:s + edge_count]
        row = {
            "split": split,
            "farm_id": int(farm_id),
            "sample_idx": int(sample_idx),
            "piece_local_id": int(local_id),
            "start_time_idx": int(piece_edges[0]["start"]),
            "end_time_idx": int(piece_edges[-1]["end"]),
        }
        for e, edge in enumerate(piece_edges):
            row[f"edge_{e}_direction"] = edge["direction"]
            row[f"edge_{e}_length"] = int(edge["length"])
            row[f"edge_{e}_avg_slope"] = float(edge["avg_slope"])
        rows.append(row)
        local_id += 1

    return rows


# =========================================================
# 4. 主生成流程
# =========================================================
def sample_stride_for_split(split: str, cfg: Config) -> int:
    if split == "train":
        return max(1, int(cfg.train_sample_stride))
    if split == "val":
        return max(1, int(cfg.val_sample_stride))
    if split == "test":
        return max(1, int(cfg.test_sample_stride))
    return 1


def generate_for_edge_and_split(edge_count: int, split: str, cfg: Config) -> Dict:
    data_dir = Path(cfg.data_dir)
    out_root = Path(cfg.output_root) / cfg.output_template.format(edge_count=edge_count)
    out_split_dir = out_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    x = load_x_lowmid(data_dir, split)
    n_samples, time_steps, num_farms, num_features = x.shape
    if num_farms < cfg.num_farms:
        raise ValueError(f"数据中风电场数量不足：需要 {cfg.num_farms}，实际 {num_farms}")

    stride = sample_stride_for_split(split, cfg)
    sample_indices = range(0, n_samples, stride)

    summary = {
        "edge_count": edge_count,
        "split": split,
        "input_shape": list(x.shape),
        "sample_stride": stride,
        "farm_counts": {},
    }

    print(f"\n===== 生成 edge_count={edge_count}, split={split}, stride={stride} =====")

    for farm_idx in range(cfg.num_farms):
        farm_id = farm_idx + 1
        out_csv = out_split_dir / f"farm_{farm_id}_pieces.csv"
        if out_csv.exists() and not cfg.overwrite:
            print(f"[已存在，跳过] {out_csv}")
            df_old = pd.read_csv(out_csv)
            summary["farm_counts"][f"farm_{farm_id}"] = int(len(df_old))
            continue

        rows = []
        for sample_idx in sample_indices:
            series = compose_lowmid_series(x[sample_idx, :, farm_idx, :], cfg.series_mode)
            rows.extend(
                build_pieces_for_sample(
                    series=series,
                    split=split,
                    farm_id=farm_id,
                    sample_idx=int(sample_idx),
                    edge_count=edge_count,
                    cfg=cfg,
                )
            )

        df = pd.DataFrame(rows)
        # 固定列顺序，保证和匹配实验脚本对齐
        base_cols = ["split", "farm_id", "sample_idx", "piece_local_id", "start_time_idx", "end_time_idx"]
        edge_cols = []
        for e in range(edge_count):
            edge_cols += [f"edge_{e}_direction", f"edge_{e}_length", f"edge_{e}_avg_slope"]
        df = df[base_cols + edge_cols]
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        summary["farm_counts"][f"farm_{farm_id}"] = int(len(df))
        print(f"farm_{farm_id}: {len(df)} pieces -> {out_csv}")

    save_json(summary, out_root / "metrics" / f"piece_generation_summary_{split}.json")
    return summary


def main() -> None:
    cfg = CFG
    save_json(asdict(cfg), Path(cfg.output_root) / "piece_library_generation_config.json")

    all_summary = []
    for edge_count in cfg.edge_counts:
        for split in cfg.splits:
            summary = generate_for_edge_and_split(edge_count, split, cfg)
            all_summary.append(summary)

    out_path = Path(cfg.output_root) / "piece_library_generation_overview.json"
    save_json(all_summary, out_path)

    print("\n===== 零件库生成完成 =====")
    print(f"总览文件: {out_path.resolve()}")
    print("下一步可运行: python piece_matching_upgrade_experiment.py")


if __name__ == "__main__":
    main()
