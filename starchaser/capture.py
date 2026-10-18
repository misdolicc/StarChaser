"""采集服务：24 小时运行，按太阳高度自动进入/退出夜间监测。

夜间主循环 (每帧):  读帧 → (FrameRing 推帧) → 实时检测 → 事件管理 (预录/续录/结束) → 快照 → 自动增益
事件写入 inbox/.tmp_<id>/，完成后原子改名为 inbox/<id>/，由处理进程 (worker) 接手。
"""
from __future__ import annotations

import collections
import json
import logging
import os
import signal
import time

import cv2
import numpy as np

from . import astro
from .calibration import CameraModel, circle_mask
from .camera import make_source
from .detector import RealtimeDetector
from .storage import Paths, event_id, free_gb, night_of, write_json_atomic
from .sysutil import cpu_temp, sd_notify

log = logging.getLogger(__name__)


def load_mask(cfg, paths: Paths, width, height):
    """检测掩膜 (全分辨率)：优先 mask.png，其次按标定生成的地平线掩膜，最后用镜头像圈。"""
    if paths.mask.exists():
        m = cv2.imread(str(paths.mask), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    if paths.calib.exists():
        try:
            model = CameraModel.load(paths.calib)
            return model.sky_mask(cfg.calibration.min_altitude_deg, width, height)
        except Exception as e:
            log.warning("标定文件无法用于掩膜: %s", e)
    r = cfg.calibration.circle_radius_px * width / cfg.camera.main_size[0] if cfg.calibration.circle_radius_px else 0
    return circle_mask(width, height, r * 0.97 if r else min(width, height) / 2 * 0.97)


class EventManager:
    def __init__(self, cfg, paths: Paths, src, scale):
        self.cfg = cfg
        self.rc = cfg.recorder
        self.paths = paths
        self.src = src
        self.scale = scale
        self.active = None
        self.recent = collections.deque()
        self.count = 0
        self.last_id = None
        self._throttle_logged = 0.0

    def update(self, res, fr):
        if res.new_confirmed:
            if self.active is None:
                self._start(res.new_confirmed, fr)
            else:
                self.active["tracks"].extend(res.new_confirmed)
        if self.active is None:
            return
        if res.active_confirmed or res.new_confirmed:
            self.active["last_activity"] = fr.ts
        a = self.active
        if fr.ts - a["last_activity"] > self.rc.post_seconds or fr.ts - a["start_ts"] > self.rc.max_event_seconds:
            self.finish(fr.ts)

    def _start(self, tracks, fr):
        while self.recent and fr.ts - self.recent[0] > 3600:
            self.recent.popleft()
        if len(self.recent) >= self.rc.max_events_per_hour:
            if fr.ts - self._throttle_logged > 300:
                log.warning("触发过于频繁 (>%d/小时)，暂停录像 (雨滴/昆虫/闪电?)", self.rc.max_events_per_hour)
                self._throttle_logged = fr.ts
            return
        self.recent.append(fr.ts)
        t0 = min(t.first.ts for t in tracks)
        eid = event_id(t0)
        tmp = self.paths.inbox / f".tmp_{eid}"
        tmp.mkdir(parents=True, exist_ok=True)
        pre = self.rc.pre_seconds + (fr.ts - t0)
        try:
            self.src.recorder.begin(tmp / "clip", pre, fr.ts)
        except RuntimeError as e:
            log.warning("无法开始录像: %s", e)
            return
        self.active = {"id": eid, "dir": tmp, "tracks": list(tracks), "start_ts": fr.ts, "trigger_ts": t0,
                       "last_activity": fr.ts, "gain": fr.meta.get("gain"), "exposure": fr.meta.get("exposure")}
        log.info("事件 %s 开始 (航迹 %d 点)", eid, len(tracks[0].points))

    def finish(self, ts):
        a, self.active = self.active, None
        if a is None:
            return
        clip = self.src.recorder.end()
        c = self.cfg
        meta = {
            "id": a["id"],
            "night": night_of(a["trigger_ts"], c.site.longitude),
            "trigger_ts": a["trigger_ts"],
            "end_ts": ts,
            "tracks": [t.to_dict() for t in a["tracks"]],
            "lores_size": list(c.camera.lores_size),
            "scale": list(self.scale),
            "camera": {"gain": a["gain"], "exposure": a["exposure"], "fps": self.src.fps},
            "clip": clip,
            "site": {"lat": c.site.latitude, "lon": c.site.longitude, "name": c.site.name},
        }
        write_json_atomic(a["dir"] / "meta.json", meta)
        final = self.paths.inbox / a["id"]
        os.replace(a["dir"], final)
        self.count += 1
        self.last_id = a["id"]
        log.info("事件 %s 完成: %d 帧, %d 条航迹", a["id"], clip.get("frames", 0), len(a["tracks"]))


class SnapshotTaker:
    """定时叠加 N 帧全分辨率图像 → 16 位 PNG (标定、云量评估、整夜合成底图)。"""

    def __init__(self, cfg, paths: Paths):
        self.interval = cfg.schedule.snapshot_interval_s
        self.n = cfg.schedule.snapshot_frames
        self.paths = paths
        self.lon = cfg.site.longitude
        self.next_t = None
        self.collecting = False
        self.acc = None
        self.count = 0
        self.t_first = 0.0
        self.gain = None
        self._restore_main = False

    def update(self, fr, src, busy: bool):
        if self.interval <= 0:
            return
        if self.next_t is None:
            self.next_t = fr.ts + 30.0
        if not self.collecting:
            if fr.ts >= self.next_t and not busy:
                self.collecting, self.acc, self.count = True, None, 0
                self.t_first, self.gain = fr.ts, fr.meta.get("gain")
                self._restore_main, src.want_main = src.want_main, True  # 采集期间才取全分辨率帧
            return
        if fr.main is None or fr.fi % 2:
            return
        g = fr.main if fr.main.ndim == 2 else cv2.cvtColor(fr.main, cv2.COLOR_BGR2GRAY)
        self.acc = g.astype(np.float32) if self.acc is None else self.acc + g
        self.count += 1
        if self.count >= self.n:
            self._save(fr.ts)
            self.collecting, self.acc = False, None
            src.want_main = self._restore_main
            self.next_t = fr.ts + self.interval

    def _save(self, t_end):
        t_mid = (self.t_first + t_end) / 2
        d = self.paths.snapshots / night_of(t_mid, self.lon)
        d.mkdir(parents=True, exist_ok=True)
        name = event_id(t_mid)
        img16 = np.clip(self.acc / self.count * 256.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(d / f"{name}.png"), img16)
        write_json_atomic(d / f"{name}.json", {"t": t_mid, "frames": self.count, "gain": self.gain})
        log.info("快照 %s/%s.png", d.name, name)


class CaptureService:
    def __init__(self, cfg, device=None, start_unix=None):
        self.cfg = cfg
        self.paths = Paths(cfg.paths.data_dir).ensure()
        self.device = device
        self.start_unix = start_unix
        self.simulate = device is not None and not str(device).isdigit()
        fps = float(cfg.camera.fps)
        self.keep = cfg.recorder.pre_seconds + cfg.camera.keyframe_interval / fps + 1.5
        self._stop = False
        self.status = {}

    def stop(self, *_):
        self._stop = True

    def is_night(self, t):
        s = self.cfg.schedule
        alt = float(astro.sun_altitude(t, self.cfg.site.latitude, self.cfg.site.longitude))
        return s.force_night or alt < s.night_sun_altitude, alt

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        sd_notify("READY=1")
        while not self._stop:
            night, alt = self.is_night(time.time())
            if self.simulate or night:
                try:
                    self.run_session()
                except Exception:
                    log.exception("夜间监测异常，30 秒后重试")
                    self._sleep(30)
                if self.simulate:
                    break
            else:
                self._write_status({"mode": "day", "sun_alt": round(alt, 2)})
                self._sleep(30)

    def _sleep(self, s):
        end = time.time() + s
        while not self._stop and time.time() < end:
            sd_notify("WATCHDOG=1")
            time.sleep(min(5.0, end - time.time()))

    def _write_status(self, extra):
        self.status.update(extra)
        self.status.update(updated=time.time(), cpu_temp=cpu_temp(), free_gb=round(free_gb(self.paths.root), 2))
        try:
            write_json_atomic(self.paths.status, self.status)
        except OSError as e:
            log.warning("写状态失败: %s", e)

    def run_session(self):
        cfg = self.cfg
        src = make_source(cfg, self.keep, self.device, self.start_unix)
        src.start()
        mw, mh = cfg.camera.main_size
        lw, lh = cfg.camera.lores_size
        if self.simulate:  # 视频文件：以文件实际分辨率为准
            fr0 = src.read()
            mh, mw = fr0.main.shape[:2]
            src.stop()
            src.start()
        mask_full = load_mask(cfg, self.paths, mw, mh)
        mask_lo = cv2.erode(cv2.resize(mask_full, (lw, lh), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
        det = RealtimeDetector(cfg.detector, (lh, lw), mask_lo, fps=src.fps)
        events = EventManager(cfg, self.paths, src, (mw / lw, mh / lh))
        snaps = SnapshotTaker(cfg, self.paths)
        cam = cfg.camera
        t_status = t_sun = t_gain = 0.0
        n_frames = dropped = 0
        first_ts = last_ts = None
        fps_win = collections.deque(maxlen=int(src.fps * 10))
        log.info("夜间监测开始")
        try:
            while not self._stop:
                fr = src.read()
                if fr is None:
                    break
                n_frames += 1
                if first_ts is None:
                    first_ts = fr.ts
                if last_ts is not None and fr.ts - last_ts > 1.5 / src.fps:
                    dropped += int(round((fr.ts - last_ts) * src.fps)) - 1
                last_ts = fr.ts
                fps_win.append(time.time())

                if fr.main is not None and src.recorder is not None:
                    src.recorder.push(fr.main, fr.ts)
                res = det.process(fr.lores, fr.fi, fr.ts)
                events.update(res, fr)
                snaps.update(fr, src, busy=events.active is not None)

                if not src.realtime:
                    continue
                now = fr.ts
                if now - t_gain > cam.gain_adjust_interval_s and events.active is None:
                    t_gain = now
                    med = float(np.median(fr.lores[::4, ::4][mask_lo[::4, ::4] > 0]))
                    new_gain = src.gain
                    if med > cam.target_background * 1.4:
                        new_gain = src.gain / 1.25
                    elif med < cam.target_background / 1.4:
                        new_gain = src.gain * 1.25
                    if abs(np.clip(new_gain, 1.0, cam.max_gain) - src.gain) > 1e-3:
                        src.set_gain(new_gain)
                        det.reset()
                        log.info("背景 %.1f → 增益 %.2f", med, src.gain)
                if now - t_sun > 60:
                    t_sun = now
                    night, alt = self.is_night(now)
                    self.status["sun_alt"] = round(alt, 2)
                    if not night and events.active is None:
                        log.info("天亮了 (太阳高度 %.1f°)，结束夜间监测", alt)
                        break
                if now - t_status > 15:
                    t_status = now
                    fps = (len(fps_win) - 1) / (fps_win[-1] - fps_win[0]) if len(fps_win) > 2 else 0
                    self._write_status({"mode": "night", "fps": round(fps, 2), "frames": n_frames,
                                        "dropped": dropped, "gain": round(src.gain, 2),
                                        "background": float(np.median(fr.lores[::8, ::8])),
                                        "events": events.count, "last_event": events.last_id,
                                        "night": night_of(now, cfg.site.longitude)})
                    sd_notify("WATCHDOG=1")
        finally:
            if events.active is not None:
                events.finish(last_ts or time.time())
            src.stop()
            log.info("夜间监测结束: %d 帧, 丢帧 %d, 事件 %d", n_frames, dropped, events.count)
            if first_ts is not None:
                self._log_session(first_ts, last_ts, n_frames, dropped, events.count)
        return events.count

    def _log_session(self, start, end, frames, dropped, n_events):
        """记录一次监测时段 (报表用来统计有效监测时长)。"""
        nd = self.paths.night_dir(night_of(start, self.cfg.site.longitude))
        nd.mkdir(parents=True, exist_ok=True)
        rec = {"start": start, "end": end, "frames": frames, "dropped": dropped, "events": n_events}
        with (nd / "sessions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
