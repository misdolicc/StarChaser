"""处理进程：消费 inbox 中的事件，生成成果文件、写数据库、更新报表；每晚结束后精修标定并收尾。

以低优先级 (nice) 运行，限制 OpenCV 线程数，不影响实时采集。
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import astro, report
from .analysis import analyze_clip, measure
from .calibration import CameraModel, calibrate, detect_stars, draw_overlay, load_catalog
from .capture import load_mask
from .storage import DB, Paths, cleanup, night_of, utc_iso, write_json_atomic
from .sysutil import sd_notify, set_nice

log = logging.getLogger(__name__)
VIDEO_EXT = (".mp4", ".h264", ".avi", ".mkv")


def remux_h264(src: Path, fps: float) -> Path:
    """裸 H.264 → MP4 (不重新编码，几乎不耗 CPU)。没有 ffmpeg 时保留 .h264，OpenCV 仍可读取。"""
    ffmpeg = shutil.which("ffmpeg")
    if src.suffix != ".h264" or not ffmpeg:
        return src
    dst = src.with_suffix(".mp4")
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", str(src),
                        "-c", "copy", str(dst)], capture_output=True)
    if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        src.unlink()
        return dst
    log.warning("ffmpeg 封装失败: %s", r.stderr.decode(errors="replace")[-300:])
    return src


def find_video(d: Path):
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in VIDEO_EXT:
            return p
    return None


# ---------------------------------------------------------------- 成果文件
def _jpg(path, img, q=90):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def _label(m: dict, tz) -> str:
    """图上标注只用 ASCII (OpenCV 内置字体不支持中文)。"""
    parts = [report.fmt_time(m["t_start"], tz, seconds=True)]
    if m.get("shower"):
        parts.append(m["shower"])
    if m.get("peak_mag") is not None:
        parts.append(f"{report.num(m['peak_mag'], '{:+.1f}')}mag")
    return "  ".join(parts)


def write_products(dest: Path, res: dict, accepted: list, model, calibrated, tz):
    st = res["stacks"]
    rx, ry, rw, rh = st.roi
    _jpg(dest / "stack.jpg", st.maxcolor)
    if not accepted:
        return
    ann = st.maxcolor.copy()
    if calibrated and model is not None:
        shifted = CameraModel.from_dict({**model.to_dict(), "x0": model.x0 - rx, "y0": model.y0 - ry,
                                         "width": rw, "height": rh})
        ann = draw_overlay(ann, shifted)
    trail = np.zeros_like(st.maxcolor)
    corridor = np.zeros(st.maxcolor.shape[:2], np.uint8)
    (dest / "frames").mkdir(exist_ok=True)
    for k, r in enumerate(accepted, 1):
        tr, m, plan = r["track"], r["measure"], r["plan"]
        pts = np.stack([np.array(tr["x"]) - rx, np.array(tr["y"]) - ry], 1)
        # 标注线画在轨迹旁 12 像素处 (平行线 + 方向箭头)，不遮挡流星本身
        d = pts[-1] - pts[0]
        nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9) * 12
        side_pts = pts + nrm
        cv2.polylines(ann, [side_pts.round().astype(np.int32)], False, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(ann, tuple(side_pts[max(0, len(pts) - 4)].round().astype(int)),
                        tuple(side_pts[-1].round().astype(int)), (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.5)
        cv2.circle(ann, tuple(pts[0].round().astype(int)), 9, (80, 220, 80), 1, cv2.LINE_AA)
        tx, ty = side_pts[:, 0].min(), side_pts[:, 1].min() - 10
        cv2.putText(ann, f"#{k} {_label(m, tz)}", (int(max(5, tx)), int(max(20, ty))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(corridor, [pts.round().astype(np.int32)], False, 255, 24)
        # 缩略图：轨迹外接框外扩
        x0, y0 = pts.min(0) - 40
        x1, y1 = pts.max(0) + 40
        side = max(x1 - x0, y1 - y0, 160)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(max(0, cy - side / 2)), int(min(rh, cy + side / 2))
        c, d = int(max(0, cx - side / 2)), int(min(rw, cx + side / 2))
        thumb = st.maxcolor[a:b, c:d]
        if thumb.size:
            s = 320 / max(thumb.shape[:2])
            _jpg(dest / f"thumb_{k}.jpg", cv2.resize(thumb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 85)
        for key in ("start", "peak", "end"):
            if plan.get(key) is not None:
                _jpg(dest / "frames" / f"m{k}_{key}.jpg", plan[key])
        if plan["crops"]:
            tiles = [cv2.putText(c_.copy(), str(i), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                     for i, c_ in plan["crops"]]
            cols = min(10, len(tiles))
            rows = math.ceil(len(tiles) / cols)
            blank = np.zeros_like(tiles[0])
            tiles += [blank] * (rows * cols - len(tiles))
            grid = np.vstack([np.hstack(tiles[r_ * cols:(r_ + 1) * cols]) for r_ in range(rows)])
            _jpg(dest / f"montage_{k}.jpg", cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    # 轨迹层只保留高于背景的流星信号，用于叠加到整夜合成图
    signal = np.clip(st.maxcolor.astype(np.float32) - st.ave[..., None], 0, 255).astype(np.uint8)
    trail[corridor > 0] = signal[corridor > 0]
    cv2.imwrite(str(dest / "trail.png"), trail)
    _jpg(dest / "annotated.jpg", ann, 88)


def meteor_row(eid, night, m, rel_dir):
    return {"id": eid, "night": night, "dir": rel_dir, **{k: m.get(k) for k in (
        "t_start", "t_end", "t_peak", "duration", "n_frames", "x0", "y0", "x1", "y1", "length_px", "alt0", "az0",
        "alt1", "az1", "ra0", "dec0", "ra1", "dec1", "length_deg", "speed_deg_s", "peak_mag", "saturated", "shower",
        "shower_dist", "calibrated")}}


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths(cfg.paths.data_dir).ensure()
        self.db = DB(self.paths.db)
        self.tz = report.get_tz(cfg.site.timezone, cfg.site.longitude)
        self._stop = False
        self._masks = {}
        self.model, self.calibrated = None, False
        self.reload_calibration()
        cv2.setNumThreads(int(cfg.analysis.threads))

    def stop(self, *_):
        self._stop = True

    def site(self):
        return {"lat": self.cfg.site.latitude, "lon": self.cfg.site.longitude, "name": self.cfg.site.name}

    def reload_calibration(self):
        self._masks.clear()
        if self.paths.calib.exists():
            self.model, self.calibrated = CameraModel.load(self.paths.calib), True
        else:  # 未标定：用配置的镜头参数做近似 (仅用于速度判别，不输出赤道坐标)
            c, w, h = self.cfg.calibration, *self.cfg.camera.main_size
            self.model = CameraModel.initial(w, h, c.fov_deg, c.circle_radius_px, c.north_angle_deg)
            self.calibrated = False

    def model_for(self, w, h):
        m = self.model
        return m if (m.width, m.height) == (w, h) else m.scaled(w, h)

    def mask_for(self, w, h):
        if (w, h) not in self._masks:
            self._masks[(w, h)] = load_mask(self.cfg, self.paths, w, h)
        return self._masks[(w, h)]

    # ------------------------------------------------------------ 主循环
    def run(self, once=False):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        set_nice(int(self.cfg.worker.nice))
        sd_notify("READY=1")
        while not self._stop:
            touched = set()
            night_now = self._is_night()
            if self.cfg.worker.process_at_night or not night_now or once:
                for item in sorted(p for p in self.paths.inbox.iterdir() if p.is_dir() and not p.name.startswith(".")):
                    if self._stop:
                        break
                    try:
                        touched.add(self.process_event(item))
                    except Exception:
                        log.exception("处理事件 %s 失败，移入 failed/", item.name)
                        failed = self.paths.root / "failed"
                        failed.mkdir(exist_ok=True)
                        shutil.move(str(item), str(failed / item.name))
                    sd_notify("WATCHDOG=1")
            for night in sorted(touched):
                report.build_night(self.paths, self.db, night, self.cfg, self.tz)
            if touched:
                report.build_index(self.paths, self.db, self.cfg, self.tz)
            if not night_now or once:
                self.finalize_due(force=once)
            if once:
                break
            sd_notify("WATCHDOG=1")
            end = time.time() + float(self.cfg.worker.poll_interval_s)
            while not self._stop and time.time() < end:
                time.sleep(0.5)

    def _is_night(self):
        alt = float(astro.sun_altitude(time.time(), self.cfg.site.latitude, self.cfg.site.longitude))
        return alt < self.cfg.schedule.night_sun_altitude

    # ------------------------------------------------------------ 单个事件
    def process_event(self, item: Path) -> str:
        meta = json.loads((item / "meta.json").read_text(encoding="utf-8"))
        night = meta["night"]
        video = find_video(item)
        if video is None:
            raise FileNotFoundError("事件目录中没有视频")
        video = remux_h264(video, float(meta["clip"].get("fps") or self.cfg.camera.fps))
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        model = self.model_for(w, h)
        res = analyze_clip(video, meta, self.cfg, model, self.calibrated, self.mask_for(w, h))
        accepted = [r for r in res["results"] if not r["reasons"]]
        kind = "meteors" if accepted else "rejected"
        dest = self.paths.night_dir(night) / kind / item.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(item), str(dest))
        write_products(dest, res, accepted, model, self.calibrated, self.tz)
        rel = dest.relative_to(self.paths.root).as_posix()
        ev = {"id": item.name, "night": night, "kind": kind, "video": video.name, "roi": list(res["stacks"].roi),
              "size": [w, h], "processed": utc_iso(time.time()), "calibrated": self.calibrated,
              "analysis": res["info"], "gain": (meta.get("camera") or {}).get("gain"),
              "fps": float(meta["clip"].get("fps") or self.cfg.camera.fps), "site": meta.get("site") or self.site(),
              "meteors": [], "rejected_tracks": []}
        for r in res["results"]:
            entry = {"track": r["track"], "measure": r["measure"], "fit": r["fit"], "reasons": r["reasons"]}
            (ev["meteors"] if not r["reasons"] else ev["rejected_tracks"]).append(entry)
        for k, entry in enumerate(ev["meteors"], 1):
            mid = item.name if len(ev["meteors"]) == 1 else f"{item.name}_{k}"
            entry["id"], entry["index"] = mid, k
            self.db.upsert_meteor(meteor_row(mid, night, entry["measure"], rel))
        write_json_atomic(dest / "event.json", ev)
        if accepted:
            log.info("流星 %s: %s", item.name, ", ".join(
                f"{e['measure'].get('shower', '?')} {e['measure'].get('speed_deg_s') or 0:.1f}°/s" for e in ev["meteors"]))
        else:
            reason = "; ".join(sorted({x for r in res["results"] for x in r["reasons"]})) or "未找到航迹"
            self.db.add_reject(item.name, night, meta.get("trigger_ts"), reason, rel)
            if not self.cfg.analysis.keep_rejected_video:
                (dest / video.name).unlink(missing_ok=True)
            log.info("排除 %s: %s", item.name, reason)
        return night

    # ------------------------------------------------------------ 每晚收尾
    def finalize_due(self, force=False):
        current = night_of(time.time(), self.cfg.site.longitude)
        for d in sorted(self.paths.nights.glob("*")):
            night = d.name
            fin, _ = self.db.night_info(night)
            if fin or (night >= current and not force):
                continue
            if any(p.is_dir() and not p.name.startswith(".") for p in self.paths.inbox.iterdir()):
                return  # 先处理完积压事件
            self.finalize_night(night)

    def finalize_night(self, night):
        log.info("整理观测夜 %s", night)
        changed = False
        if self.cfg.calibration.auto_refine:
            changed = self.refine_calibration(night)
        if changed:
            self.reprocess_astrometry(night)
        report.build_composite(self.paths, self.db, night)
        report.build_night(self.paths, self.db, night, self.cfg, self.tz)
        report.build_index(self.paths, self.db, self.cfg, self.tz)
        cleanup(self.paths, self.cfg.storage)
        self.db.set_night(night, finalized=True)

    def refine_calibration(self, night, image=None, t_unix=None, blind=None, gain=1.0) -> bool:
        """用当夜星最多的快照 (或指定图像) 标定。成功且不劣于现有标定时更新 calib.json。"""
        c = self.cfg.calibration
        site = self.site()
        if image is None:
            best = None
            for png in sorted((self.paths.snapshots / night).glob("*.png")):
                img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                img = img.astype(np.float32) / (256.0 if img.dtype == np.uint16 else 1.0)
                n = len(detect_stars(img, self.mask_for(img.shape[1], img.shape[0]), max_stars=400)["x"])
                if best is None or n > best[0]:
                    meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
                    best = (n, img, meta["t"], meta.get("gain") or 1.0)
            if best is None:
                log.info("观测夜 %s 没有快照，跳过标定", night)
                return False
            _, image, t_unix, gain = best
        h, w = image.shape[:2]
        init = self.model_for(w, h) if self.calibrated else CameraModel.initial(
            w, h, c.fov_deg, c.circle_radius_px * w / self.cfg.camera.main_size[0], c.north_angle_deg)
        if blind is None:
            blind = not self.calibrated
        mask = self.mask_for(w, h)
        model, rep = calibrate(image, t_unix, site["lat"], site["lon"], init, mask=mask, blind=blind,
                               max_mag=c.max_mag, gain=gain, min_stars=c.min_stars, max_rms_arcmin=c.max_rms_arcmin)
        log.info("标定结果: %s", {k: v for k, v in rep.items() if k != "matches"})
        check = draw_overlay(image, model, t_unix, site["lat"], site["lon"],
                             stars=detect_stars(image, mask), catalog=load_catalog())
        cv2.imwrite(str(self.paths.calib_dir / f"check_{night}.jpg"), check, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not rep["ok"]:
            return False
        if self.calibrated and self.model.rms_arcmin and model.rms_arcmin > self.model.rms_arcmin * 1.5:
            log.info("新标定 rms %.1f′ 劣于现有 %.1f′，保留现有标定", model.rms_arcmin, self.model.rms_arcmin)
            return False
        hist = self.paths.calib_dir / "history"
        hist.mkdir(exist_ok=True)
        if self.paths.calib.exists():
            shutil.copy2(self.paths.calib, hist / f"calib_before_{night}.json")
        model.save(self.paths.calib)
        if not self.paths.mask.exists():  # 生成初始掩膜，用户可用画图软件涂黑树木/建筑/灯光
            cv2.imwrite(str(self.paths.mask), model.sky_mask(c.min_altitude_deg))
        self.reload_calibration()
        log.info("标定已更新: 匹配 %d 颗, rms %.2f′", model.n_stars, model.rms_arcmin)
        return True

    def reprocess_astrometry(self, night):
        """标定更新后，用 event.json 中保存的像素轨迹重新计算天体测量 (无需重新解码视频)。"""
        for evf in sorted((self.paths.night_dir(night) / "meteors").glob("*/event.json")):
            ev = json.loads(evf.read_text(encoding="utf-8"))
            model = self.model_for(*ev["size"])
            for entry in ev["meteors"]:
                old = entry["measure"]
                m = measure(entry["track"], model, ev["site"], ev.get("gain"), ev["fps"], self.calibrated)
                m["saturated"] = old.get("saturated", 0)
                entry["measure"] = m
                self.db.upsert_meteor(meteor_row(entry["id"], night, m, evf.parent.relative_to(self.paths.root).as_posix()))
            ev["calibrated"] = self.calibrated
            write_json_atomic(evf, ev)
