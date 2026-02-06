# Autonomous AI Data Platform for Self-Driving

This repo provides a runnable scaffold for an autonomous data platform workflow:

1. Collect raw data (simulated by default)
2. Clean and align data
3. Generate labels
4. Train models (placeholder)
5. Generate reports
6. Visualize and query in a dashboard

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the pipeline once:

```bash
python -c "from src.scheduler.tasks import task_scheduler, define_tasks; task_scheduler(define_tasks())"
```

3. Launch the dashboard:

```bash
streamlit run dashboards/streamlit_app.py
```

## Key Modules

- `src/data_pipeline/collector.py`: data collection (synthetic by default)
- `src/data_pipeline/cleaner.py`: sync, anomaly detection, labels
- `src/models/`: model training placeholders
- `src/scheduler/`: task orchestration
- `dashboards/streamlit_app.py`: Streamlit UI

## Next Steps

- Replace synthetic data generation with CARLA/KITTI/nuScenes integrations.
- Swap LLM stubs with real client calls.
- Implement real model training pipelines.
