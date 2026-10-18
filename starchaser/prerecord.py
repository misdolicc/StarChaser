"""预录像：在内存环形缓冲中保留最近数秒视频，触发时连同"过去"一起写入文件。

- EncodedRing  : 树莓派方案。挂在 picamera2 硬件 H.264 编码器后面，缓存已编码的帧
                 (8 Mbps × 5 s ≈ 5 MB 内存)。写盘从关键帧开始，保证片段可独立解码。
- FrameRing    : 通用方案 (USB 摄像头 / 视频文件仿真)。缓存原始帧，触发时用 OpenCV 写 MJPG。
两者接口相同: push(frame, ts) / begin(path, pre_seconds, now_ts) / end() -> clip_info
"""
from __future__ import annotations

import collections
import logging
import threading
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

try:  # 仅树莓派上可用
    from picamera2.outputs import Output as _PiOutput
except Exception:  # pragma: no cover
    class _PiOutput:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            self.recording = False

        def start(self):
            self.recording = True

        def stop(self):
            self.recording = False


class EncodedRing(_PiOutput):
    """picamera2 Output：编码帧环形缓冲 + 按需落盘。

    时间戳: 编码器给出的是相对编码开始的微秒数；begin() 时用当前帧的墙钟时间做锚点换算。
    """

    def __init__(self, keep_seconds: float, fps: float):
        super().__init__()
        self.keep_us = int(keep_seconds * 1e6)
        self.fps = fps
        self._ring = collections.deque()
        self._lock = threading.Lock()
        self._file = None
        self._pts: list[int] = []
        self._latest_ts = None
        self._anchor = None
        self.path = None

    # 编码器线程回调 (兼容不同 picamera2 版本的签名)
    def outputframe(self, frame, keyframe=True, timestamp=None, *args, **kwargs):
        ts = int(timestamp or 0)
        with self._lock:
            self._latest_ts = ts
            if self._file is not None:
                self._file.write(frame)
                self._pts.append(ts)
                return
            self._ring.append((bytes(frame), bool(keyframe), ts))
            while self._ring and ts - self._ring[0][2] > self.keep_us:
                self._ring.popleft()

    def push(self, frame, ts):  # 编码器自行推帧，这里无事可做
        pass

    def begin(self, path, pre_seconds: float, now_ts: float):
        with self._lock:
            if self._latest_ts is None:
                raise RuntimeError("编码器尚未输出任何帧")
            self.path = Path(path).with_suffix(".h264")
            self._anchor = (self._latest_ts, now_ts)
            cutoff = self._latest_ts - int(pre_seconds * 1e6)
            frames = list(self._ring)
            # 找到不晚于 cutoff 的最后一个关键帧；若没有则用最早的关键帧
            start = None
            for i, (_, key, ts) in enumerate(frames):
                if key and (ts <= cutoff or start is None):
                    start = i
                if ts > cutoff and start is not None:
                    break
            self._file = open(self.path, "wb", buffering=1 << 20)
            self._pts = []
            for data, _, ts in frames[start or 0:]:
                self._file.write(data)
                self._pts.append(ts)
            self._ring.clear()

    def end(self) -> dict:
        with self._lock:
            f, self._file = self._file, None
            pts, self._pts = self._pts, []
        if f is None:
            return {}
        f.close()
        a_ts, a_wall = self._anchor
        frame_ts = [a_wall + (p - a_ts) / 1e6 for p in pts]
        return {"file": self.path.name, "fps": self.fps, "frames": len(pts), "codec": "h264",
                "start_ts": frame_ts[0] if frame_ts else a_wall, "frame_ts": [round(t, 4) for t in frame_ts]}


class FrameRing:
    """原始帧环形缓冲，触发时用 OpenCV 写入 MJPG/AVI。"""

    def __init__(self, keep_seconds: float, fps: float, quality: int = 92):
        self.keep = keep_seconds
        self.fps = fps
        self.quality = quality
        self._ring = collections.deque()
        self._writer = None
        self._ts: list[float] = []
        self.path = None

    def push(self, frame, ts):
        if self._writer is not None:
            self._write(frame, ts)
            return
        self._ring.append((frame, ts))
        while self._ring and ts - self._ring[0][1] > self.keep:
            self._ring.popleft()

    def _write(self, frame, ts):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._writer.write(frame)
        self._ts.append(ts)

    def begin(self, path, pre_seconds: float, now_ts: float):
        if not self._ring:
            raise RuntimeError("缓冲区为空")
        self.path = Path(path).with_suffix(".avi")
        h, w = self._ring[0][0].shape[:2]
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        self._writer.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
        self._ts = []
        for frame, ts in self._ring:
            if ts >= now_ts - pre_seconds - 1e-6:
                self._write(frame, ts)
        self._ring.clear()

    def end(self) -> dict:
        if self._writer is None:
            return {}
        self._writer.release()
        self._writer = None
        ts, self._ts = self._ts, []
        return {"file": self.path.name, "fps": self.fps, "frames": len(ts), "codec": "mjpg",
                "start_ts": ts[0] if ts else None, "frame_ts": [round(t, 4) for t in ts]}
