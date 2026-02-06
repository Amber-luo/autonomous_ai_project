from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_pipeline.cleaner import detect_anomalies, llm_generate_labels, synchronize_sensors
from src.utils.storage import read_parquet, write_parquet


def build_processed_dataset(raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    camera = read_parquet(raw_dir / "camera.parquet")
    lidar = read_parquet(raw_dir / "lidar.parquet")
    imu = read_parquet(raw_dir / "imu.parquet")

    merged = synchronize_sensors([camera, lidar, imu])
    merged = detect_anomalies(merged)
    merged = llm_generate_labels(merged, llm_model=None)

    write_parquet(merged, processed_dir / "dataset.parquet")
    return merged
