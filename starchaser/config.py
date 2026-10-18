"""配置：内置默认值 + 用户 TOML 覆盖 (Python 3.11+ 自带 tomllib，无需额外依赖)。"""
from __future__ import annotations

import copy
import logging
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULTS: dict = {
    "site": {
        "name": "StarChaser",
        "latitude": 31.23,       # 纬度，北正
        "longitude": 121.47,     # 经度，东正
        "elevation": 10.0,       # 海拔 m
        "timezone": "",          # 报表显示时区 (IANA，如 "Asia/Shanghai")，空 = 系统本地时区
    },
    "paths": {
        "data_dir": "~/starchaser-data",
    },
    "camera": {
        "source": "picamera2",   # picamera2 | opencv (USB 摄像头或视频文件)
        "device": 0,             # opencv: 设备号或视频文件路径
        "main_size": [1920, 1080],
        "lores_size": [640, 360],
        "fps": 25,
        "bitrate": 8_000_000,
        "keyframe_interval": 25,  # H.264 I 帧间隔 (帧)，决定预录起点的对齐精度
        "linear_gamma": True,     # 修改 ISP 调优为线性 gamma，便于测光
        "tuning_file": "",        # 为空则按传感器型号自动选择
        "initial_gain": 8.0,
        "max_gain": 16.0,
        "target_background": 35,  # 夜空背景目标亮度 (0-255)，自动增益据此调节
        "gain_adjust_interval_s": 60,
        "noise_reduction": "fast",  # off | fast | hq
        "colour_gains": [0.0, 0.0],  # 固定白平衡增益 [R, B]；0 = 不设置 (黑白传感器)
        "buffer_count": 6,
    },
    "schedule": {
        "night_sun_altitude": -10.0,  # 太阳高度角低于此值开始监测
        "force_night": False,         # 调试用：忽略太阳高度一直监测
        "snapshot_interval_s": 600,   # 夜间叠加快照间隔 (用于标定/云量)
        "snapshot_frames": 32,
    },
    "detector": {
        "k_sigma": 4.5,               # 差分阈值 = k·σ
        "min_delta": 10.0,            # 最小亮度差 (DN)
        "bg_alpha": 0.05,             # 背景 EMA 系数
        "noise_alpha": 0.02,
        "warmup_s": 2.0,
        "min_area": 1,                # 斑点最小像素数
        "max_blobs": 60,              # 单帧斑点过多视为全局变化 (闪电/车灯/云)
        "global_change_fraction": 0.02,
        "gate_px": 10.0,              # 新航迹关联门限 (lores 像素)
        "max_gap_frames": 2,
        "min_frames": 4,
        "min_displacement_px": 2.0,
        "min_speed_px": 0.15,         # lores 像素/帧；全天镜头 0.5°/px 时约 2°/s
        "max_track_s": 5.0,           # 超长航迹 (飞机/卫星) 丢弃
        "linear_rms_px": 1.2,
    },
    "recorder": {
        "pre_seconds": 3.0,           # 预录时长
        "post_seconds": 2.0,          # 航迹结束后继续录制
        "max_event_seconds": 15.0,
        "max_events_per_hour": 120,   # 触发风暴保护
    },
    "analysis": {
        "k_sigma": 4.0,
        "min_delta": 8.0,
        "min_frames": 4,
        "min_length_px": 12.0,        # 全分辨率像素
        "min_speed_deg_s": 1.5,
        "max_speed_deg_s": 90.0,
        "max_duration_s": 8.0,
        "max_tracks": 3,
        "max_linear_rms_px": 2.5,
        "keep_rejected_video": False,
        "threads": 1,                 # OpenCV 线程数，限制以免抢占采集进程
    },
    "calibration": {
        "fov_deg": 185.0,             # 镜头视场 (直径方向)
        "circle_radius_px": 0,        # 鱼眼像圈半径；0 = min(宽,高)/2
        "north_angle_deg": -1,        # 正北在图像中相对"上方"的顺时针角度；-1 = 未知 (自动搜索)
        "min_altitude_deg": 5.0,      # 低于此高度角的区域不参与检测
        "max_mag": 5.0,               # 标定使用的星表极限星等
        "max_rms_arcmin": 8.0,
        "min_stars": 20,
        "auto_refine": True,          # 每晚结束后用最佳快照自动精修标定
    },
    "storage": {
        "min_free_gb": 2.0,
        "keep_video_days": 90,
        "keep_snapshot_days": 30,
    },
    "worker": {
        "process_at_night": True,     # False = 只在白天处理事件
        "poll_interval_s": 5.0,
        "nice": 15,
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
    },
}


class Cfg(dict):
    """支持属性访问的字典：cfg.camera.fps"""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, key, value):
        self[key] = value


def _merge(base: dict, override: dict, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in base:
            log.warning("未知配置项 %s%s (已忽略)", prefix, k)
            continue
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise ValueError(f"配置项 {prefix}{k} 应为表 (table)")
            out[k] = _merge(base[k], v, f"{prefix}{k}.")
        else:
            out[k] = v
    return out


def _to_cfg(d: dict) -> Cfg:
    return Cfg({k: _to_cfg(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    data = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path).expanduser()
        with p.open("rb") as f:
            data = _merge(data, tomllib.load(f))
    if overrides:
        data = _merge(data, overrides)
    cfg = _to_cfg(data)
    cfg.paths.data_dir = str(Path(cfg.paths.data_dir).expanduser())
    return cfg
