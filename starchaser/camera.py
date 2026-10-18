"""帧源：picamera2 (CSI 相机，硬件 H.264) 与 OpenCV (USB 摄像头 / 视频文件)。"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .prerecord import EncodedRing, FrameRing

log = logging.getLogger(__name__)


@dataclass
class Frame:
    fi: int
    ts: float                      # 曝光时刻的墙钟时间 (Unix 秒)
    lores: np.ndarray              # 检测用灰度图 (uint8)
    main: np.ndarray | None = None  # 全分辨率图 (快照 / FrameRing 录像用)，按需提供
    meta: dict = field(default_factory=dict)


class PiCameraSource:
    """CSI 相机：main 流 → 硬件 H.264 → EncodedRing；lores 流 → 检测。"""

    realtime = True

    def __init__(self, cfg, keep_seconds: float):
        self.c = cfg.camera
        self.fps = float(self.c.fps)
        self.recorder = EncodedRing(keep_seconds, self.fps)
        self.picam2 = None
        self.gain = float(self.c.initial_gain)
        self.fi = 0
        self._offset = None
        self._offset_t = 0.0
        self.want_main = False

    def _tuning(self):
        from picamera2 import Picamera2

        if not self.c.linear_gamma and not self.c.tuning_file:
            return None
        try:
            name = self.c.tuning_file or f"{Picamera2.global_camera_info()[0]['Model']}.json"
            tuning = Picamera2.load_tuning_file(name)
            if self.c.linear_gamma:
                algo = Picamera2.find_tuning_algo(tuning, "rpi.contrast")
                algo["ce_enable"] = 0
                algo["gamma_curve"] = [0, 0, 65535, 65535]
            return tuning
        except Exception as e:  # 调优文件不可用时退回默认
            log.warning("无法加载/修改调优文件: %s", e)
            return None

    def start(self):
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder

        self.picam2 = Picamera2(tuning=self._tuning())
        frame_us = int(1e6 / self.fps)
        config = self.picam2.create_video_configuration(
            main={"size": tuple(self.c.main_size), "format": "YUV420"},
            lores={"size": tuple(self.c.lores_size), "format": "YUV420"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            buffer_count=int(self.c.buffer_count),
        )
        self.picam2.configure(config)
        encoder = H264Encoder(bitrate=int(self.c.bitrate), repeat=True, iperiod=int(self.c.keyframe_interval))
        self.picam2.start_recording(encoder, self.recorder)
        self._apply_controls()
        self.fi = 0
        log.info("相机启动: main=%s lores=%s fps=%.1f", self.c.main_size, self.c.lores_size, self.fps)

    def _apply_controls(self):
        ctrl = {"AeEnable": False, "AwbEnable": False, "ExposureTime": int(1e6 / self.fps),
                "AnalogueGain": float(self.gain)}
        nr = {"off": 0, "fast": 1, "hq": 2}.get(str(self.c.noise_reduction).lower())
        if nr is not None:
            ctrl["NoiseReductionMode"] = nr
        r, b = self.c.colour_gains
        if r > 0 and b > 0:
            ctrl["ColourGains"] = (float(r), float(b))
        try:
            self.picam2.set_controls(ctrl)
        except Exception as e:
            log.warning("设置相机控制失败 (%s)，尝试最小控制集", e)
            self.picam2.set_controls({"AeEnable": False, "ExposureTime": ctrl["ExposureTime"],
                                      "AnalogueGain": ctrl["AnalogueGain"]})

    def set_gain(self, gain):
        self.gain = float(np.clip(gain, 1.0, self.c.max_gain))
        self.picam2.set_controls({"AnalogueGain": self.gain})

    def _wall_time(self, sensor_ns):
        """传感器时间戳 → 墙钟。取 (接收时刻 - 传感器时刻) 的滑动最小值作为偏移，与时钟源无关。"""
        now = time.time()
        off = now - sensor_ns / 1e9
        if self._offset is None or off < self._offset or now - self._offset_t > 600:
            self._offset, self._offset_t = off, now
        return sensor_ns / 1e9 + self._offset

    def read(self) -> Frame | None:
        req = self.picam2.capture_request()
        try:
            lw, lh = self.c.lores_size
            lores = req.make_array("lores")[:lh, :lw].copy()
            main = None
            if self.want_main:
                mw, mh = self.c.main_size
                main = req.make_array("main")[:mh, :mw].copy()
            md = req.get_metadata()
        finally:
            req.release()
        ts = self._wall_time(md.get("SensorTimestamp", time.time_ns()))
        self.fi += 1
        return Frame(self.fi, ts, lores, main,
                     {"gain": md.get("AnalogueGain", self.gain), "exposure": md.get("ExposureTime"),
                      "lux": md.get("Lux"), "temp": md.get("SensorTemperature")})

    def stop(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop_recording()
            finally:
                self.picam2.close()
                self.picam2 = None


class OpenCVSource:
    """USB 摄像头或视频文件。视频文件模式下按文件帧率生成时间戳 (仿真)。"""

    def __init__(self, cfg, keep_seconds: float, device=None, start_unix: float | None = None):
        self.c = cfg.camera
        dev = self.c.device if device is None else device
        self.is_file = isinstance(dev, str) and not str(dev).isdigit()
        self.device = dev if self.is_file else int(dev)
        self.realtime = not self.is_file
        self.start_unix = start_unix
        self.fps = float(self.c.fps)
        self.gain = float(self.c.initial_gain)
        self.recorder = None
        self.keep_seconds = keep_seconds
        self.cap = None
        self.fi = 0
        self.want_main = True

    def start(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源 {self.device}")
        if self.is_file:
            side = Path(str(self.device)).with_suffix(".json")
            if side.exists():
                meta = json.loads(side.read_text(encoding="utf-8"))
                self.start_unix = self.start_unix or meta.get("start_unix")
                self.fps = float(meta.get("fps", self.fps))
            else:
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            if self.start_unix is None:
                self.start_unix = time.time()
        else:
            w, h = self.c.main_size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.recorder = FrameRing(self.keep_seconds, self.fps)
        self.fi = 0

    def set_gain(self, gain):
        self.gain = float(np.clip(gain, 1.0, self.c.max_gain))
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_GAIN, self.gain)

    def read(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        ts = self.start_unix + self.fi / self.fps if self.is_file else time.time()
        self.fi += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lw, lh = self.c.lores_size
        lores = cv2.resize(gray, (lw, lh), interpolation=cv2.INTER_AREA)
        return Frame(self.fi, ts, lores, img, {"gain": self.gain})

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def make_source(cfg, keep_seconds, device=None, start_unix=None):
    if device is not None or cfg.camera.source == "opencv":
        return OpenCVSource(cfg, keep_seconds, device, start_unix)
    return PiCameraSource(cfg, keep_seconds)
