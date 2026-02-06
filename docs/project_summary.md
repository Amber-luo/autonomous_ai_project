# Autonomous AI Data Platform 项目文档汇总

> 本文档整理：需求文档要点、开发文档要点、实际问题与解决方案（按你的实际构建过程总结）。

---

## 1. 项目需求文档要点

**项目名称**：大模型驱动的自治智能数据平台（Autonomous AI Data Platform for Self-Driving）

**项目背景**：
- 自动驾驶研发涉及海量多模态数据（Camera/LiDAR/IMU/CAN/日志/仿真/实车）。
- 传统流程人工依赖重：采集、清洗、标注、训练、分析、报告。

**项目目的**：
1. 自动化数据管道：采集、清洗、标准化、存储。
2. 智能分析与模型训练：目标检测、时序异常检测。
3. 大模型应用：标注、异常分析、报告、问答。
4. 自治调度：自动触发数据处理/训练/报告。
5. 可视化与问答：仪表盘 + 自然语言查询。

**核心问题与痛点**：
- 数据量大、格式多 → 清洗困难。
- 标注成本高 → 需要 LLM 辅助。
- 异常发现慢 → 时序检测 + 解释不足。
- 训练调度复杂 → 自动化不足。
- 报告滞后 → 需要自动生成 + 可视化。

---

## 2. 开发文档要点

**数据源**：
- CARLA 仿真（Camera/LiDAR/IMU）
- KITTI / nuScenes（公开多模态）
- 仿真日志（CSV/JSON）

**技术模块**：
1. 数据采集层：CARLA + 数据集下载
2. 清洗处理：时间对齐、坐标标准化、异常检测、压缩存储、LLM 标签
3. 模型训练：目标检测（YOLO/Detectron）+ 异常检测（LSTM/Transformer）
4. 调度系统：Prefect / schedule
5. 可视化与问答：Streamlit/Dash + LLM

**数据结构示例**：
- Camera: timestamp, camera_id, image_path, width, height
- LiDAR: timestamp, lidar_id, pointcloud_path, num_points
- IMU: timestamp, speed, accel_x/y/z, steering, brake

**功能流程**：
1. 采集 → raw
2. 清洗 → processed
3. LLM 标签
4. 训练 → 评估 → LLM 报告
5. 自动报告
6. 调度执行
7. 可视化与问答

---

## 3. 实际实施步骤（你已完成的主链路）

### 3.1 数据采集（CARLA）
- CARLA Server 运行确认
- Python API 版本匹配（0.9.16 + Python 3.12）
- 采集回调保存图片/点云 + parquet 元数据
- 输出：`data/raw/{run_id}/camera|lidar|imu` + parquet + meta.json

### 3.2 数据清洗
- 读取 raw parquet
- 时间对齐（merge_asof）
- 异常检测（IsolationForest）
- 标签生成（占位规则）
- 输出 processed parquet

### 3.3 训练占位 + 报告
- detector 与 anomaly 训练使用占位逻辑生成指标
- 生成报告保存到 `reports/*.txt`

### 3.4 可视化仪表盘
- Streamlit 选择 processed 文件
- 数据预览 + 异常统计图
- 报告展示

### 3.5 LLM 问答（开源模型 Ollama）
- 本地 Ollama API 调用
- Streamlit 内嵌问答 + 数据上下文摘要

---

## 4. 你提过的问题与解决方案（实际过程回顾）

### Q1：`camera_rows` 为什么不会清空？
- **原因**：Python 列表是可变对象，函数内 `append` 会修改同一个对象。

### Q2：为什么先保存图片再写元数据？
- **原因**：保证元数据中的路径真实存在，避免写入无效路径。

### Q3：元数据保存在哪里？
- **位置**：`data/raw/{run_id}/camera.parquet` 等 parquet 文件。

### Q4：CARLA 无法启动（DirectX Runtime）
- **解决**：安装 DirectX Runtime 后再启动。

### Q5：`pip install carla==0.9.16` 找不到
- **原因**：PyPI 无对应版本
- **解决**：使用 CARLA 自带的 `whl/egg` 安装

### Q6：Python 版本不匹配
- **原因**：Python 3.14 无对应 wheel
- **解决**：安装 Python 3.12 并创建 venv

### Q7：PowerShell 激活 venv 报错
- **原因**：执行策略限制
- **解决**：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 或用 `activate.bat`

### Q8：Streamlit 报 `ModuleNotFoundError: src`
- **原因**：运行目录导致导入失败
- **解决**：在 `streamlit_app.py` 中追加项目根目录到 `sys.path`

### Q9：`ModuleNotFoundError: sklearn`
- **原因**：错误安装包 `sklearn`
- **解决**：安装 `scikit-learn`

### Q10：Ollama 端口占用
- **原因**：Ollama 已启动
- **解决**：不用重复 `ollama serve`，直接 `ollama ps/run`

---

## 5. 当前已实现成果

- ✅ CARLA 采集 + parquet 元数据
- ✅ 清洗 + 异常检测 + 标签生成
- ✅ 训练占位 + 报告生成
- ✅ Streamlit 可视化
- ✅ Ollama LLM 问答

---

## 6. 建议后续扩展路线

1. **异常可视化增强**：异常样本筛选 + 图片预览
2. **报告升级**：HTML/PDF 输出 + 图表
3. **真实训练接入**：YOLO/Detectron/LSTM/Transformer
4. **调度自动化**：Prefect / schedule
5. **数据集接入**：KITTI / nuScenes

---

## 7. 文件索引（已实现）

- `src/data_pipeline/collector.py` 采集（CARLA）
- `src/data_pipeline/cleaner.py` 清洗
- `src/models/train_detector.py` 占位训练
- `src/models/train_anomaly.py` 占位训练
- `src/models/train_runner.py` 训练 + 报告
- `src/llm/qna_ai.py` Ollama 问答
- `dashboards/streamlit_app.py` 仪表盘
- `reports/*.txt` 报告输出

---

## 8. 使用流程（简版）

1. 采集：`collect_carla_data`
2. 清洗：`clean_run(data/raw/{run_id})`
3. 训练报告：`run_training_and_report`
4. Dashboard：`streamlit run dashboards/streamlit_app.py`
5. QnA：仪表盘内输入问题

---

**文档完成。**
