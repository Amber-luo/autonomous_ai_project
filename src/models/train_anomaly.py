from __future__ import annotations

"""
占位训练：时序/异常检测
- 不训练真实模型，只做简单统计
"""

from typing import Any

import pandas as pd


# ==========================================================
# 1. 训练入口（占位）
# ==========================================================

def train_anomaly_model(df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, float]:
    """
    传入清洗后的 DataFrame，返回一个伪指标字典。
    这里用 anomaly 列统计异常比例，作为占位结果。
    """
    if config is None:
        config = {}

    total = len(df)
    if total == 0:
        return {"anomaly_ratio": 0.0, "loss": 1.0, "samples": 0}

    if "anomaly" in df.columns:
        anomaly_ratio = float((df["anomaly"] == -1).mean())
    else:
        anomaly_ratio = 0.0

    loss = float(min(1.0, anomaly_ratio + 0.1))
    return {"anomaly_ratio": anomaly_ratio, "loss": loss, "samples": float(total)}


# ==========================================================
# 2. 评估入口（占位）
# ==========================================================

def evaluate_model(model_metrics: dict[str, float], test_data: pd.DataFrame | None = None) -> dict[str, float]:
    """
    真实评估时应在 test_data 上计算指标。
    目前直接返回训练阶段的占位指标。
    """
    return model_metrics
