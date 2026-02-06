from __future__ import annotations

"""
数据采集（CARLA）入门版：
- 目标：让你能一步一步看懂并跑起来
- 仅实现“采集模块”，其它模块不改

运行前提：
1) CARLA Server 已启动（通常是 CarlaUE4.exe）
2) 本机安装了 CARLA Python API（carla 包）
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import log_event
from src.utils.storage import write_parquet


# ==========================================================
# 1. 入口函数：你会从这里开始调用
# ==========================================================

def collect_carla_data(config: dict[str, Any], output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    用 CARLA 采集数据，并保存到磁盘。

    参数：
    - config: 配置字典（主机、端口、采集时长、帧率等）
    - output_dir: 原始数据保存目录（例如 data/raw）

    返回：
    - 一个包含 camera/lidar/imu 的 DataFrame 字典
    """

    # 1) 处理输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) 若你想先不连 CARLA，可用 simulate=True 跳过真实采集
    if config.get("simulate", False):
        log_event("collector", "simulate=True, skip CARLA collection")
        return _generate_empty_frames(output_dir)

    # 3) 动态导入 CARLA
    try:
        import carla  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CARLA Python API 不可用，请先安装 carla 包，或设置 simulate=True。"
        ) from exc

    # 4) 读取常用配置（给你一个可调的入口）
    host = config.get("host", "localhost")
    port = int(config.get("port", 2000))
    timeout = float(config.get("timeout", 10.0))
    fps = float(config.get("fps", 10.0))
    duration_s = float(config.get("duration_s", 10.0))
    run_id = str(config.get("run_id", time.strftime("%Y%m%d_%H%M%S")))

    # 5) 创建本次采集的目录结构
    run_dir = output_dir / run_id
    (run_dir / "camera").mkdir(parents=True, exist_ok=True)
    (run_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (run_dir / "imu").mkdir(parents=True, exist_ok=True)

    # 6) 连接 CARLA
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    # 7) 选择地图（可选）
    if config.get("map"):
        world = client.load_world(config["map"])
    else:
        world = client.get_world()

    # 8) 强制同步模式（重要：保证传感器数据按帧对齐）
    original_settings = world.get_settings()
    sync_settings = carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=1.0 / fps,
        max_substep_delta_time=1.0 / fps,
        max_substeps=1,
    )
    world.apply_settings(sync_settings)

    # 9) 准备采集容器
    camera_rows: list[dict[str, Any]] = []
    lidar_rows: list[dict[str, Any]] = []
    imu_rows: list[dict[str, Any]] = []

    vehicle = None
    sensors = []

    try:
        # ------------------------------------------------------
        # Block A: 生成车辆
        # ------------------------------------------------------
        vehicle = _spawn_vehicle(world, config.get("vehicle_filter", "vehicle.*"))

        # ------------------------------------------------------
        # Block B: 挂载传感器（相机、激光雷达、IMU）
        # ------------------------------------------------------
        sensors = _spawn_sensors(world, vehicle, config)

        # ------------------------------------------------------
        # Block C: 注册传感器回调（回调里负责落盘 + 记元数据）
        # ------------------------------------------------------
        for sensor in sensors:
            if sensor.type_id.startswith("sensor.camera"):
                sensor.listen(
                    lambda image, s=sensor: _handle_camera(image, s, run_dir, camera_rows)
                )
            elif sensor.type_id.startswith("sensor.lidar"):
                sensor.listen(
                    lambda pc, s=sensor: _handle_lidar(pc, s, run_dir, lidar_rows)
                )
            elif sensor.type_id == "sensor.other.imu":
                sensor.listen(lambda imu, s=sensor: _handle_imu(imu, s, imu_rows))

        # ------------------------------------------------------
        # Block D: 推进世界帧（同步模式需要手动 tick）
        # ------------------------------------------------------
        frames = int(duration_s * fps)
        for _ in range(frames):
            world.tick()

        # 等待一点时间，确保回调写完
        time.sleep(0.2)

    finally:
        # ------------------------------------------------------
        # Block E: 清理资源（非常重要）
        # ------------------------------------------------------
        for sensor in sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)

    # ----------------------------------------------------------
    # Block F: 保存元数据表
    # ----------------------------------------------------------
    camera_df = pd.DataFrame(camera_rows)
    lidar_df = pd.DataFrame(lidar_rows)
    imu_df = pd.DataFrame(imu_rows)

    write_parquet(camera_df, run_dir / "camera.parquet")
    write_parquet(lidar_df, run_dir / "lidar.parquet")
    write_parquet(imu_df, run_dir / "imu.parquet")

    meta = {
        "run_id": run_id,
        "frames": int(duration_s * fps),
        "fps": fps,
        "host": host,
        "port": port,
        "map": config.get("map"),
        "vehicle_filter": config.get("vehicle_filter", "vehicle.*"),
        "sensors": [s.type_id for s in sensors],
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_event("collector", "CARLA collection completed", run_id=run_id)

    return {"camera": camera_df, "lidar": lidar_df, "imu": imu_df}


# ==========================================================
# 2. 车辆生成
# ==========================================================

def _spawn_vehicle(world, vehicle_filter: str):
    """
    根据过滤条件生成一辆车
    - vehicle_filter: "vehicle.*" 表示随机一辆
    """
    blueprint_library = world.get_blueprint_library()
    vehicles = blueprint_library.filter(vehicle_filter)
    if not vehicles:
        raise RuntimeError(f"No vehicles found for filter: {vehicle_filter}")

    vehicle_bp = vehicles[0]
    spawn_points = world.get_map().get_spawn_points()
    transform = spawn_points[0] if spawn_points else None

    return world.spawn_actor(vehicle_bp, transform)


# ==========================================================
# 3. 传感器生成（相机 / LiDAR / IMU）
# ==========================================================

def _spawn_sensors(world, vehicle, config: dict[str, Any]):
    """
    挂载传感器到车辆上。你可以从这里开始扩展多相机 / 多雷达。
    """
    blueprint_library = world.get_blueprint_library()

    # 相机
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(config.get("camera_width", 1280)))
    camera_bp.set_attribute("image_size_y", str(config.get("camera_height", 720)))
    camera_bp.set_attribute("fov", str(config.get("camera_fov", 90)))

    # LiDAR
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", str(config.get("lidar_channels", 32)))
    lidar_bp.set_attribute("range", str(config.get("lidar_range", 50.0)))
    lidar_bp.set_attribute("points_per_second", str(config.get("lidar_pps", 56000)))

    # IMU
    imu_bp = blueprint_library.find("sensor.other.imu")

    sensors = []
    sensors.append(
        world.spawn_actor(camera_bp, _sensor_transform(1.5, 0.0, 2.4), attach_to=vehicle)
    )
    sensors.append(
        world.spawn_actor(lidar_bp, _sensor_transform(0.0, 0.0, 2.2), attach_to=vehicle)
    )
    sensors.append(
        world.spawn_actor(imu_bp, _sensor_transform(0.0, 0.0, 2.0), attach_to=vehicle)
    )

    return sensors


def _sensor_transform(x: float, y: float, z: float):
    import carla  # type: ignore

    return carla.Transform(carla.Location(x=x, y=y, z=z))


# ==========================================================
# 4. 传感器回调（真实数据落盘）
# ==========================================================

def _handle_camera(image, sensor, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """
    相机回调：保存图片 + 写元数据
    """
    file_path = run_dir / "camera" / f"{image.frame:06d}.png"
    image.save_to_disk(str(file_path))
    rows.append(
        {
            "timestamp": image.timestamp,
            "camera_id": sensor.id,
            "image_path": str(file_path),
            "width": image.width,
            "height": image.height,
            "frame": image.frame,
        }
    )


def _handle_lidar(pointcloud, sensor, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """
    LiDAR 回调：保存点云 + 写元数据
    """
    file_path = run_dir / "lidar" / f"{pointcloud.frame:06d}.ply"
    pointcloud.save_to_disk(str(file_path))
    rows.append(
        {
            "timestamp": pointcloud.timestamp,
            "lidar_id": sensor.id,
            "pointcloud_path": str(file_path),
            "num_points": len(pointcloud),
            "frame": pointcloud.frame,
        }
    )


def _handle_imu(imu, sensor, rows: list[dict[str, Any]]) -> None:
    """
    IMU 回调：只写元数据（IMU 数据量小）
    """
    rows.append(
        {
            "timestamp": imu.timestamp,
            "imu_id": sensor.id,
            "accel_x": imu.accelerometer.x,
            "accel_y": imu.accelerometer.y,
            "accel_z": imu.accelerometer.z,
            "gyro_x": imu.gyroscope.x,
            "gyro_y": imu.gyroscope.y,
            "gyro_z": imu.gyroscope.z,
        }
    )


# ==========================================================
# 5. simulate 模式：如果你没有 CARLA 环境
# ==========================================================

def _generate_empty_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    这里只返回空表，目的是让你在没有 CARLA 时也能跑流程。
    """
    camera_df = pd.DataFrame(columns=["timestamp", "camera_id", "image_path", "width", "height", "frame"])
    lidar_df = pd.DataFrame(columns=["timestamp", "lidar_id", "pointcloud_path", "num_points", "frame"])
    imu_df = pd.DataFrame(columns=["timestamp", "imu_id", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

    write_parquet(camera_df, output_dir / "camera.parquet")
    write_parquet(lidar_df, output_dir / "lidar.parquet")
    write_parquet(imu_df, output_dir / "imu.parquet")

    return {"camera": camera_df, "lidar": lidar_df, "imu": imu_df}
