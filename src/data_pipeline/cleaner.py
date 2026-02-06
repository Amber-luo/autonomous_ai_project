from __future__ import annotations

"""
数据清洗模块（入门版）
目标：让你能一步一步看懂并跑通

流程：
1) 读取 raw 目录下的 camera/lidar/imu parquet
2) 按 timestamp 对齐（merge_asof）
3) 字段名整理（frame_x/frame_y -> camera_frame/lidar_frame）
4) 做一个最简单的异常检测（IsolationForest）
5) 可选：生成标签（暂时用规则/占位）
6) 保存到 processed 目录
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.llm import label_ai
from src.utils.storage import read_parquet, write_parquet

# ods层清洗

# ==========================================================
# [ADDED] 1. 读取原始数据
# ==========================================================

def load_raw_run(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    [ADDED] 给定一次采集的 run_dir，读取 camera/lidar/imu 的 parquet。
    """
    run_dir = Path(run_dir)

    camera_df = read_parquet(run_dir / "camera.parquet")
    lidar_df = read_parquet(run_dir / "lidar.parquet")
    imu_df = read_parquet(run_dir / "imu.parquet")

    return {"camera": camera_df, "lidar": lidar_df, "imu": imu_df}


# ==========================================================
# 2. 时间同步（对齐 timestamp）
# ==========================================================

def synchronize_sensors(sensor_dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple sensor dataframes on timestamp using merge_asof.
    """
    dfs = list(sensor_dfs)
    if not dfs:
        return pd.DataFrame()
    
    #camera作为基准表
    merged = dfs[0].sort_values("timestamp")
    for df in dfs[1:]:
        merged = pd.merge_asof(
            merged,
            df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )
    return merged


# ==========================================================
# [ADDED] 3. 字段名整理
# ==========================================================

def rename_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    [ADDED] 合并后会出现 frame_x / frame_y，改成更清晰的名字。
    """
    rename_map = {}
    if "frame_x" in df.columns:
        rename_map["frame_x"] = "camera_frame"
    if "frame_y" in df.columns:
        rename_map["frame_y"] = "lidar_frame"

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ==========================================================
# 4. 坐标标准化（可选）
# ==========================================================

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in ["x", "y", "z"]:
        if col in normalized.columns:
            normalized[col] = normalized[col] - normalized[col].mean()
    return normalized


# ==========================================================
# 5. 异常检测（最小可用版本）
# ==========================================================

def detect_anomalies(df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """
    Use IsolationForest to mark anomalies. Non-float columns are ignored.
    """
    numeric_df = df.select_dtypes(include=["float", "int"])
    if numeric_df.empty:
        df["anomaly"] = 1
        return df

    clf = IsolationForest(contamination=min(0.1, 1.0 / max(1, len(df))))
    df["anomaly"] = clf.fit_predict(numeric_df)
    return df


# ==========================================================
# [ADDED] 6. 异常统计
# ==========================================================

def summarize_anomalies(df: pd.DataFrame) -> dict[str, float | int]:
    """
    [ADDED] 返回异常数量与比例，方便快速检查。
    """
    if "anomaly" not in df.columns:
        return {"total": len(df), "anomaly_count": 0, "anomaly_ratio": 0.0}

    total = int(len(df))
    anomaly_count = int((df["anomaly"] == -1).sum())
    ratio = float(anomaly_count / total) if total > 0 else 0.0
    return {"total": total, "anomaly_count": anomaly_count, "anomaly_ratio": ratio}


# ==========================================================
# 7. 压缩保存（可选）
# ==========================================================

def compress_data(df: pd.DataFrame, file_path: str) -> None:
    df.to_parquet(file_path, index=False, compression="snappy")


# ==========================================================
# 8. 标签生成（占位）
# ==========================================================

def llm_generate_labels(df: pd.DataFrame, llm_model=None) -> pd.DataFrame:
    labeled_df = label_ai.generate_labels(df, llm_model)
    return labeled_df


# ==========================================================
# [ADDED] 9. 主流程：清洗 + 保存
# ==========================================================

def clean_run(run_dir: str | Path, output_dir: str | Path) -> Path:
    """
    [ADDED] 清洗一次采集结果，并保存到 processed 目录。
    返回保存后的 parquet 路径。
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 读取 raw
    raw = load_raw_run(run_dir)

    # Step 2: 时间对齐
    merged = synchronize_sensors([raw["camera"], raw["lidar"], raw["imu"]])

    # Step 3: 字段名整理
    merged = rename_frame_columns(merged)

    # Step 4: 异常检测
    merged = detect_anomalies(merged)

    # Step 5: 生成标签（占位）
    merged = llm_generate_labels(merged, llm_model=None)

    # Step 6: 保存 processed 数据
    out_path = output_dir / f"{run_dir.name}_dataset.parquet"
    write_parquet(merged, out_path)

    return out_path
