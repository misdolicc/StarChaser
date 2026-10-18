"""端到端：合成视频 → 采集 (实时检测 + 预录) → 处理 (分析 + 天体测量 + 归属) → 报表。"""
import json
import os
import time

import cv2
import numpy as np

from conftest import LAT, LON
from starchaser import astro, synth
from starchaser.capture import CaptureService
from starchaser.config import load_config
from starchaser.storage import DB, Paths, cleanup, event_id, night_of, write_json_atomic
from starchaser.worker import Worker


def make_cfg(tmp_path, **cal):
    return load_config(overrides={
        "paths": {"data_dir": str(tmp_path / "data")},
        "site": {"latitude": LAT, "longitude": LON},
        "camera": {"main_size": [960, 540], "lores_size": [480, 270]},
        "schedule": {"force_night": True, "snapshot_interval_s": 0},
        "calibration": {"circle_radius_px": 270, **cal}})


def gem_scene(tmp_path, model, t_obs):
    gem = (112.0 + 1.0 * (float(astro.solar_longitude_j2000(t_obs)) - 262.2), 33.0)
    start = [float(v) for v in astro.apparent_altaz_to_j2000(50.0, 210.0, t_obs + 4, LAT, LON)]
    met = synth.SynthMeteor(4.0, 0.6, *start, *gem, speed_deg_s=20.0, peak_mag=0.0)
    plane = synth.SynthPlane(250, 380, 5.0, -1.5)
    video = tmp_path / "night.avi"
    synth.make_video(video, model, t_obs, 12.0, 25, LAT, LON, [met], [plane])
    return video, start


def test_end_to_end(tmp_path, true_model, t_obs):
    cfg = make_cfg(tmp_path, auto_refine=False)
    paths = Paths(cfg.paths.data_dir).ensure()
    model = true_model
    model.zp, model.zp_gain = 9.4, 8.0
    model.save(paths.calib)  # 假设已标定 (标定本身在 test_calibration 中验证)
    video, start = gem_scene(tmp_path, model, t_obs)

    CaptureService(cfg, device=str(video)).run()
    events = [p for p in paths.inbox.iterdir() if p.is_dir()]
    assert len(events) >= 1

    Worker(cfg).run(once=True)
    rows = DB(paths.db).meteors()
    assert len(rows) == 1
    m = rows[0]
    assert m["shower"] == "GEM"
    assert abs(m["speed_deg_s"] - 20.0) < 2.0
    assert abs(m["duration"] - 0.6) < 0.15
    assert -1.5 < m["peak_mag"] < 1.5
    # 第一帧质心是曝光中点，比真实起点晚约 半帧 × 角速度 ≈ 0.4°
    assert astro.angular_sep(m["ra0"], m["dec0"], *start) < 1.0

    night_dir = paths.nights / m["night"]
    assert (night_dir / "report.html").exists() and (night_dir / "events.csv").exists()
    ev_dir = paths.root / m["dir"]
    for f in ("annotated.jpg", "stack.jpg", "thumb_1.jpg", "montage_1.jpg", "frames/m1_peak.jpg", "event.json"):
        assert (ev_dir / f).exists(), f
    ev = json.loads((ev_dir / "event.json").read_text(encoding="utf-8"))
    assert ev["meteors"][0]["measure"]["track_radec"]
    assert (paths.root / "index.html").exists()


def test_uncalibrated_then_backfilled(tmp_path, true_model, t_obs):
    """首次标定之前的流星：只有像素数据；当夜收尾时自动盲标定，并回填天球坐标与流星雨。"""
    cfg = make_cfg(tmp_path)
    paths = Paths(cfg.paths.data_dir).ensure()
    video, start = gem_scene(tmp_path, true_model, t_obs)
    night = night_of(t_obs, LON)
    snap_dir = paths.snapshots / night
    snap_dir.mkdir(parents=True)
    snap = synth.make_snapshot(true_model, t_obs, LAT, LON)
    cv2.imwrite(str(snap_dir / f"{event_id(t_obs)}.png"), (snap * 256).astype(np.uint16))
    write_json_atomic(snap_dir / f"{event_id(t_obs)}.json", {"t": t_obs, "gain": 8.0})

    CaptureService(cfg, device=str(video)).run()
    w = Worker(cfg)
    assert not w.calibrated
    for item in sorted(p for p in paths.inbox.iterdir() if p.is_dir()):
        w.process_event(item)
    (m,) = DB(paths.db).meteors()
    assert m["calibrated"] == 0 and m["ra0"] is None and m["az0"] is None
    assert m["speed_deg_s"] is not None  # 近似模型仍可用于角速度判别

    w.finalize_night(night)  # 盲标定 → 重算
    assert w.calibrated and (paths.root / "mask.png").exists()
    (m,) = DB(paths.db).meteors()
    assert m["calibrated"] == 1 and m["shower"] == "GEM"
    assert astro.angular_sep(m["ra0"], m["dec0"], *start) < 1.0
    assert (paths.nights / night / "composite.jpg").exists()


def test_cleanup_priorities(tmp_path):
    cfg = make_cfg(tmp_path)
    paths = Paths(cfg.paths.data_dir).ensure()
    rej = paths.nights / "2026-01-01" / "rejected" / "a"
    old = paths.nights / "2026-01-01" / "meteors" / "b"
    new = paths.nights / "2026-03-01" / "meteors" / "c"
    for d in (rej, old, new):
        d.mkdir(parents=True)
        (d / "clip.mp4").write_bytes(b"x")
        (d / "stack.jpg").write_bytes(b"x")
    t_old = time.time() - 200 * 86400
    os.utime(old / "clip.mp4", (t_old, t_old))
    cleanup(paths, cfg.storage)
    assert not (rej / "clip.mp4").exists()                  # 被拒视频
    assert not (old / "clip.mp4").exists()                  # 超过 keep_video_days
    assert (new / "clip.mp4").exists()
    assert all((d / "stack.jpg").exists() for d in (rej, old, new))  # 图片永远保留
