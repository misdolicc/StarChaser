"""命令行入口:  python -m starchaser <命令> [-c 配置文件]

    capture     采集服务 (24 小时运行；--video 时用视频文件仿真)
    worker      处理服务 (分析事件、写报表、每晚收尾；--once 处理完即退出)
    calibrate   天体测量标定 (用某夜最佳快照或指定图片)
    report      重新生成报表
    reprocess   用当前标定重算某夜的天体测量
    web         局域网报表服务
    synth       生成合成测试视频
    demo        在 PC 上跑完整演示：合成夜空 → 采集 → 标定 → 分析 → 报表
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CONFIGS = ("starchaser.toml", "~/.config/starchaser.toml", "/etc/starchaser.toml")


def parse_time(s: str) -> float:
    """ISO 时间 → Unix 秒；不带时区时按 UTC。"""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def find_config(path):
    if path:
        return path
    if os.environ.get("STARCHASER_CONFIG"):
        return os.environ["STARCHASER_CONFIG"]
    for p in DEFAULT_CONFIGS:
        if Path(p).expanduser().exists():
            return str(Path(p).expanduser())
    return None


def utf8_console():
    """Windows 控制台默认编码无法输出中文。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def setup_logging(verbose):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


# ---------------------------------------------------------------- 命令
def cmd_capture(cfg, a):
    from .capture import CaptureService
    CaptureService(cfg, device=a.video, start_unix=parse_time(a.start) if a.start else None).run()


def cmd_worker(cfg, a):
    from .worker import Worker
    Worker(cfg).run(once=a.once)


def cmd_calibrate(cfg, a):
    import cv2
    import numpy as np
    from .storage import night_of
    from .worker import Worker

    w = Worker(cfg)
    if a.image:
        img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"无法读取 {a.image}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / (256.0 if img.dtype == np.uint16 else 1.0)
        side = Path(a.image).with_suffix(".json")
        t = parse_time(a.time) if a.time else json.loads(side.read_text(encoding="utf-8"))["t"]
        night = night_of(t, cfg.site.longitude)
        ok = w.refine_calibration(night, image=img, t_unix=t, blind=a.blind or None, gain=a.gain)
    else:
        night = a.night or sorted(p.name for p in w.paths.snapshots.glob("*"))[-1]
        ok = w.refine_calibration(night, blind=a.blind or None)
    print(f"核对图: {w.paths.calib_dir / f'check_{night}.jpg'}")
    print("标定成功" if ok else "标定未更新 (见上方日志)")
    sys.exit(0 if ok else 1)


def cmd_report(cfg, a):
    from . import report
    from .storage import DB, Paths
    paths = Paths(cfg.paths.data_dir).ensure()
    db = DB(paths.db)
    tz = report.get_tz(cfg.site.timezone, cfg.site.longitude)
    nights = [a.night] if a.night else [n["night"] for n in db.nights()]
    for n in nights:
        report.build_composite(paths, db, n)
        print(report.build_night(paths, db, n, cfg, tz))
    report.build_index(paths, db, cfg, tz)
    print(paths.root / "index.html")


def cmd_reprocess(cfg, a):
    from . import report
    from .worker import Worker
    w = Worker(cfg)
    w.reprocess_astrometry(a.night)
    report.build_night(w.paths, w.db, a.night, cfg, w.tz)


def cmd_web(cfg, a):
    from .web import serve
    serve(cfg.paths.data_dir, a.host or cfg.web.host, a.port or cfg.web.port)


def demo_scene(t0, duration, lat, lon, n_meteors, seed=7):
    """随机生成一组流星：约 70% 来自当时最强的活跃流星雨，其余为偶发流星。"""
    import numpy as np
    from . import astro
    from .showers import active_showers
    from .synth import SynthMeteor

    rng = np.random.default_rng(seed)
    cands = []
    for s in active_showers(t0):
        alt, _ = astro.j2000_to_apparent_altaz(s["ra_now"], s["dec_now"], t0, lat, lon)
        if alt > 10:
            cands.append(s)
    main = max(cands, key=lambda s: s["zhr"]) if cands else None
    meteors = []
    slots = np.linspace(4, duration - 4, n_meteors)
    for k in range(n_meteors):
        for _ in range(200):
            t = float(slots[k] + rng.uniform(-1, 1))
            alt0, az0 = rng.uniform(30, 75), rng.uniform(0, 360)
            ra0, dec0 = [float(v) for v in astro.apparent_altaz_to_j2000(alt0, az0, t0 + t, lat, lon)]
            if main is not None and rng.random() < 0.7:
                rra, rdec, v = main["ra_now"], main["dec_now"], main["v"]
            else:  # 偶发：随机辐射点
                rra, rdec, v = rng.uniform(0, 360), float(np.degrees(np.arcsin(rng.uniform(-1, 1)))), rng.uniform(20, 70)
            dist = float(astro.angular_sep(ra0, dec0, rra, rdec))
            if not 20 < dist < 80:
                continue
            m = SynthMeteor(t, float(rng.uniform(0.35, 0.9)), ra0, dec0, rra, rdec,
                            float(v / 66 * 30 * math.sin(math.radians(dist)) + 5), float(rng.uniform(-1.0, 2.0)))
            ra1, dec1 = astro.vec_to_radec(m.vec_at(m.duration))
            alt1, _ = astro.j2000_to_apparent_altaz(ra1, dec1, t0 + t + m.duration, lat, lon)
            if alt1 > 15:
                meteors.append(m)
                break
    return meteors, main


def cmd_synth(cfg, a):
    import calendar
    from .calibration import CameraModel
    from .synth import SynthPlane, make_video

    t0 = parse_time(a.start) if a.start else calendar.timegm((2026, 12, 13, 18, 0, 0))
    lat, lon = cfg.site.latitude, cfg.site.longitude
    w, h = cfg.camera.main_size
    model = CameraModel(w, h, w / 2 + 4.3, h / 2 - 2.2, h / 2, math.radians(92), -0.03, 0.02,
                        tilt_x=1.2, tilt_y=-0.6, roll=203.0)
    meteors, main = demo_scene(t0, a.duration, lat, lon, a.meteors)
    side = make_video(a.output, model, t0, a.duration, cfg.camera.fps, lat, lon, meteors,
                      [SynthPlane(w * 0.25, h * 0.7, 6.0, -2.0, 800)])
    model.save(Path(a.output).with_suffix(".truth_calib.json"))
    print(f"写入 {a.output}: {side['frames']} 帧, {len(meteors)} 颗流星"
          + (f" (主要流星雨 {main['name']} {main['code']})" if main else ""))
    return model, side, meteors


def cmd_demo(cfg, a):
    """PC 端完整演示，不需要相机。"""
    import calendar
    import cv2
    import numpy as np
    from . import synth
    from .capture import CaptureService
    from .storage import Paths, event_id, night_of, write_json_atomic
    from .worker import Worker

    log = logging.getLogger("demo")
    root = Path(a.data).resolve()
    cfg.paths.data_dir = str(root / "data")
    cfg.camera.main_size, cfg.camera.lores_size = [1280, 720], [640, 360]
    cfg.calibration.circle_radius_px, cfg.calibration.north_angle_deg = 360, -1
    cfg.schedule.force_night = True
    cfg.site.timezone = cfg.site.timezone or "Asia/Shanghai"
    paths = Paths(cfg.paths.data_dir).ensure()
    t0 = parse_time(a.start) if a.start else calendar.timegm((2026, 12, 13, 18, 0, 0))
    lat, lon = cfg.site.latitude, cfg.site.longitude
    night = night_of(t0, lon)

    log.info("1/4 生成合成夜空视频 (%d 秒, %d 颗流星)…", a.duration, a.meteors)
    a.output, a.start = str(root / "synth.avi"), datetime.fromtimestamp(t0, timezone.utc).isoformat()
    model, side, meteors = cmd_synth(cfg, a)
    # 模拟采集服务在当晚拍下的一张叠加快照 (标定用)
    snap = synth.make_snapshot(model, t0, lat, lon)
    sd = paths.snapshots / night
    sd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sd / f"{event_id(t0)}.png"), (snap * 256).astype(np.uint16))
    write_json_atomic(sd / f"{event_id(t0)}.json", {"t": t0, "frames": 32, "gain": cfg.camera.initial_gain})

    log.info("2/4 采集服务读取视频：实时检测 + 预录像…")
    CaptureService(cfg, device=a.output).run()

    log.info("3/4 盲标定 (自动认星，不提供镜头方向)…")
    w = Worker(cfg)
    w.refine_calibration(night, blind=True)

    log.info("4/4 处理事件并生成报表…")
    w.run(once=True)
    print("\n真值对照:")
    for i, m in enumerate(meteors, 1):
        print(f"  流星 {i}: t=+{m.t0:.1f}s 角速度 {m.speed_deg_s:.1f}°/s 辐射点 ({m.radiant_ra:.1f}, {m.radiant_dec:.1f})")
    print(f"\n报表: {paths.night_dir(night) / 'report.html'}\n索引: {paths.root / 'index.html'}")


# ---------------------------------------------------------------- 入口
def main(argv=None):
    utf8_console()
    ap = argparse.ArgumentParser(prog="starchaser", description="树莓派全天流星监测")
    ap.add_argument("-c", "--config", help="TOML 配置文件")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="采集服务")
    p.add_argument("--video", help="用视频文件仿真 (同名 .json 侧车提供开始时间)")
    p.add_argument("--start", help="视频开始时间 (ISO 8601, 默认 UTC)")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("worker", help="处理服务")
    p.add_argument("--once", action="store_true", help="处理完积压事件后退出")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("calibrate", help="天体测量标定")
    p.add_argument("--night", help="使用该观测夜星最多的快照 (默认最近一夜)")
    p.add_argument("--image", help="指定图片 (灰度/16 位 PNG)")
    p.add_argument("--time", help="图片拍摄时间 (ISO 8601)；默认读取同名 .json")
    p.add_argument("--gain", type=float, default=1.0, help="图片对应的模拟增益 (测光零点用)")
    p.add_argument("--blind", action="store_true", help="忽略现有标定，从头盲搜索")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("report", help="重新生成报表")
    p.add_argument("--night")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("reprocess", help="用当前标定重算天体测量")
    p.add_argument("--night", required=True)
    p.set_defaults(func=cmd_reprocess)

    p = sub.add_parser("web", help="局域网报表服务")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_web)

    for name, fn in (("synth", cmd_synth), ("demo", cmd_demo)):
        p = sub.add_parser(name, help="生成合成测试视频" if name == "synth" else "PC 端完整演示")
        if name == "synth":
            p.add_argument("output", help="输出 .avi")
        else:
            p.add_argument("--data", default="demo", help="演示数据目录")
        p.add_argument("--start", help="模拟的观测时间 (ISO 8601 UTC)，默认 2026-12-13T18:00Z (双子座流星雨)")
        p.add_argument("--duration", type=float, default=40.0)
        p.add_argument("--meteors", type=int, default=5)
        p.set_defaults(func=fn)

    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    from .config import load_config
    cfg = load_config(find_config(a.config))
    t = time.time()
    a.func(cfg, a)
    logging.getLogger(__name__).debug("完成，用时 %.1fs", time.time() - t)


if __name__ == "__main__":
    main()
