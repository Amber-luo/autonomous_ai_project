from __future__ import annotations

"""
占位训练：目标检测
- 不训练真实模型，只用简单统计生成“可跑通”的指标
- 让你先把完整流程打通，后续再替换成 YOLO/Detectron2
"""

from typing import Any

import pandas as pd


# ==========================================================
# 1. 训练入口（占位）
# ==========================================================

def train_detector(df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, float]:
    """
    传入清洗后的 DataFrame，返回一个伪指标字典。
    这里只做最基础的统计，模拟训练输出。
    """
    if config is None:
        config = {}

    total = len(df)
    if total == 0:
        return {"accuracy": 0.0, "loss": 1.0, "samples": 0}

    # 如果有 label，计算“多数类占比”作为占位 accuracy
    if "label" in df.columns:
        majority_ratio = df["label"].value_counts(normalize=True).iloc[0]
        accuracy = float(majority_ratio)
    else:
        accuracy = 0.5  # 没有标签时给一个固定占位值

    loss = float(1.0 - min(0.99, accuracy))
    return {"accuracy": accuracy, "loss": loss, "samples": float(total)}
