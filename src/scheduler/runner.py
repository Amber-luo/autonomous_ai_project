from __future__ import annotations

from pathlib import Path

from src.data_pipeline.collector import collect_carla_data
from src.data_pipeline.labeler import build_processed_dataset
from src.llm import report_ai
from src.models.train_anomaly import evaluate_model, train_anomaly_model
from src.models.train_detector import train_detector
from src.utils.logger import log_event


def run_task(task_name: str) -> None:
    if task_name == "collect_data":
        collect_carla_data({"simulate": True}, "data/raw")
        log_event("task", "collect_data completed")
        return

    if task_name == "clean_data":
        build_processed_dataset("data/raw", "data/processed")
        log_event("task", "clean_data completed")
        return

    if task_name == "train_detector":
        metrics = train_detector(None, None, {})
        _write_report(metrics, "detector_report.txt")
        log_event("task", "train_detector completed")
        return

    if task_name == "train_anomaly":
        metrics = train_anomaly_model(None, None, {})
        metrics = evaluate_model(metrics, None)
        _write_report(metrics, "anomaly_report.txt")
        log_event("task", "train_anomaly completed")
        return

    if task_name == "generate_report":
        _write_report({"status": "ok"}, "summary_report.txt")
        log_event("task", "generate_report completed")
        return

    raise ValueError(f"Unknown task: {task_name}")


def _write_report(metrics: dict, filename: str) -> None:
    report = report_ai.generate_report(metrics, logs=None, llm_model=None)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / filename, "w", encoding="utf-8") as f:
        f.write(report)
