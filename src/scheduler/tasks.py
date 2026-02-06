from __future__ import annotations

from typing import Iterable

import schedule

from src.scheduler.runner import run_task


def define_tasks() -> list[str]:
    return ["collect_data", "clean_data", "train_detector", "train_anomaly", "generate_report"]


def task_scheduler(task_list: Iterable[str]) -> None:
    for task in task_list:
        run_task(task)


def run_daily(task_list: Iterable[str], time_str: str = "02:00") -> None:
    for task in task_list:
        schedule.every().day.at(time_str).do(run_task, task)
