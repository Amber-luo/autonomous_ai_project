from __future__ import annotations

"""
训练 + 报告 主流程（占位版）
- 读取 processed 数据
- 生成占位训练指标
- 输出报告到 reports/ 目录
"""

from pathlib import Path

import pandas as pd

from src.llm import report_ai
from src.models.train_anomaly import evaluate_model, train_anomaly_model
from src.models.train_detector import train_detector
from src.utils.storage import read_parquet


def run_training_and_report(processed_path: str | Path, reports_dir: str | Path) -> Path:
    processed_path = Path(processed_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(processed_path)

    # 1) 占位目标检测指标
    det_metrics = train_detector(df, config={})

    # 2) 占位异常检测指标
    ano_metrics = train_anomaly_model(df, config={})
    ano_metrics = evaluate_model(ano_metrics, test_data=None)

    # 3) 合并指标
    metrics = {
        "detector_accuracy": det_metrics.get("accuracy"),
        "detector_loss": det_metrics.get("loss"),
        "anomaly_ratio": ano_metrics.get("anomaly_ratio"),
        "anomaly_loss": ano_metrics.get("loss"),
        "samples": det_metrics.get("samples"),
    }

    # 4) 生成报告
    report_text = report_ai.generate_report(metrics, logs=None, llm_model=None)

    # 5) 保存报告
    report_path = reports_dir / (processed_path.stem + "_report.txt")
    report_path.write_text(report_text, encoding="utf-8")

    return report_path
