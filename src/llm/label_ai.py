from __future__ import annotations

from typing import Any

import pandas as pd


def generate_labels(df: pd.DataFrame, llm_model: Any | None = None) -> pd.DataFrame:
    """
    Simple label generator.
    If an LLM client is provided, it can be called in the future.
    Currently adds a binary label based on speed and anomaly fields if present.
    """
    labeled = df.copy()
    if "label" in labeled.columns:
        return labeled

    label = []
    for _, row in labeled.iterrows():
        is_anomaly = bool(row.get("anomaly", 1) == -1)
        speed = float(row.get("speed", 0.0))
        label.append("anomaly" if is_anomaly or speed > 30.0 else "normal")

    labeled["label"] = label
    return labeled
