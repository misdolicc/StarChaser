"""数据目录布局、"观测夜"划分、SQLite 数据库、磁盘清理。

data_dir/
  inbox/<event_id>/            采集服务写入的待处理事件 (clip + meta.json)；写完才从 .tmp_ 改名
  nights/<YYYY-MM-DD>/
      meteors/<event_id>/      确认的流星：视频、叠加图、关键帧、event.json
      rejected/<event_id>/     被分析排除的候选 (默认只保留叠加图)
      report.html  events.csv  summary.json  composite.jpg
  snapshots/<YYYY-MM-DD>/      夜间叠加快照 (16 位 PNG + json)
  calibration/                 calib.json、历史版本、核对图
  mask.png                     可选：人工编辑的检测掩膜 (白=有效)，全分辨率
  meteors.db  status.json  index.html
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class Paths:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.inbox = self.root / "inbox"
        self.nights = self.root / "nights"
        self.snapshots = self.root / "snapshots"
        self.calib_dir = self.root / "calibration"
        self.calib = self.calib_dir / "calib.json"
        self.mask = self.root / "mask.png"
        self.db = self.root / "meteors.db"
        self.status = self.root / "status.json"

    def ensure(self):
        for p in (self.inbox, self.nights, self.snapshots, self.calib_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def night_dir(self, night):
        return self.nights / night


def night_of(t_unix: float, lon: float) -> str:
    """观测夜标识：按地方平太阳时减 12 小时取日期 (傍晚到次日清晨属于同一夜)，与系统时区无关。"""
    solar = datetime.fromtimestamp(t_unix, timezone.utc) + timedelta(hours=lon / 15.0 - 12.0)
    return solar.strftime("%Y-%m-%d")


def utc_iso(t_unix: float) -> str:
    return datetime.fromtimestamp(t_unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def event_id(t_unix: float) -> str:
    return datetime.fromtimestamp(t_unix, timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- 数据库
SCHEMA = """
CREATE TABLE IF NOT EXISTS meteors (
    id TEXT PRIMARY KEY,
    night TEXT NOT NULL,
    t_start REAL, t_end REAL, t_peak REAL,
    duration REAL, n_frames INTEGER,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, length_px REAL,
    alt0 REAL, az0 REAL, alt1 REAL, az1 REAL,
    ra0 REAL, dec0 REAL, ra1 REAL, dec1 REAL,
    length_deg REAL, speed_deg_s REAL,
    peak_mag REAL, saturated INTEGER,
    shower TEXT, shower_dist REAL,
    calibrated INTEGER,
    dir TEXT
);
CREATE INDEX IF NOT EXISTS meteors_night ON meteors(night);
CREATE TABLE IF NOT EXISTS rejects (
    id TEXT PRIMARY KEY, night TEXT, t_start REAL, reason TEXT, dir TEXT
);
CREATE TABLE IF NOT EXISTS nights (
    night TEXT PRIMARY KEY,
    finalized INTEGER DEFAULT 0,
    info TEXT
);
"""

METEOR_COLS = ("id", "night", "t_start", "t_end", "t_peak", "duration", "n_frames", "x0", "y0", "x1", "y1",
               "length_px", "alt0", "az0", "alt1", "az1", "ra0", "dec0", "ra1", "dec1", "length_deg",
               "speed_deg_s", "peak_mag", "saturated", "shower", "shower_dist", "calibrated", "dir")


class DB:
    def __init__(self, path):
        self.con = sqlite3.connect(str(path), timeout=30)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def upsert_meteor(self, row: dict):
        vals = [row.get(c) for c in METEOR_COLS]
        self.con.execute(f"INSERT OR REPLACE INTO meteors ({','.join(METEOR_COLS)}) "
                         f"VALUES ({','.join('?' * len(METEOR_COLS))})", vals)
        self.con.commit()

    def add_reject(self, eid, night, t_start, reason, d):
        self.con.execute("INSERT OR REPLACE INTO rejects VALUES (?,?,?,?,?)", (eid, night, t_start, reason, d))
        self.con.commit()

    def meteors(self, night=None):
        if night:
            cur = self.con.execute("SELECT * FROM meteors WHERE night=? ORDER BY t_start", (night,))
        else:
            cur = self.con.execute("SELECT * FROM meteors ORDER BY t_start")
        return [dict(r) for r in cur]

    def rejects(self, night):
        return [dict(r) for r in self.con.execute("SELECT * FROM rejects WHERE night=? ORDER BY t_start", (night,))]

    def nights(self):
        q = """SELECT n AS night, (SELECT COUNT(*) FROM meteors m WHERE m.night=n) AS meteors,
                      (SELECT COUNT(*) FROM rejects r WHERE r.night=n) AS rejects
               FROM (SELECT night AS n FROM meteors UNION SELECT night FROM rejects UNION SELECT night FROM nights)
               ORDER BY n DESC"""
        return [dict(r) for r in self.con.execute(q)]

    def night_info(self, night):
        r = self.con.execute("SELECT finalized, info FROM nights WHERE night=?", (night,)).fetchone()
        return (bool(r["finalized"]), json.loads(r["info"] or "{}")) if r else (False, {})

    def set_night(self, night, finalized=None, info=None):
        old_fin, old_info = self.night_info(night)
        fin = old_fin if finalized is None else finalized
        merged = {**old_info, **(info or {})}
        self.con.execute("INSERT OR REPLACE INTO nights VALUES (?,?,?)", (night, int(fin), json.dumps(merged)))
        self.con.commit()

    def close(self):
        self.con.close()


# ---------------------------------------------------------------- 磁盘清理
def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _videos(d: Path):
    return [p for p in d.glob("*") if p.suffix.lower() in (".mp4", ".h264", ".avi", ".mkv")]


def cleanup(paths: Paths, cfg_storage, now=None):
    """按优先级释放空间：被拒视频 → 过期快照 → 过期流星视频 → 最旧流星视频。叠加图与 JSON 始终保留。"""
    now = now or time.time()
    removed = []

    def rm(p: Path):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(str(p))
        except OSError as e:
            log.warning("删除失败 %s: %s", p, e)

    for v in paths.nights.glob("*/rejected/*/*"):
        if v.suffix.lower() in (".mp4", ".h264", ".avi"):
            rm(v)
    snap_age = cfg_storage.keep_snapshot_days * 86400
    for d in sorted(paths.snapshots.glob("*")):
        if d.is_dir() and now - d.stat().st_mtime > snap_age:
            rm(d)
    vid_age = cfg_storage.keep_video_days * 86400
    meteor_videos = sorted((v for e in paths.nights.glob("*/meteors/*") for v in _videos(e)),
                           key=lambda p: p.stat().st_mtime)
    for v in meteor_videos:
        if now - v.stat().st_mtime > vid_age:
            rm(v)
    for v in meteor_videos:
        if free_gb(paths.root) >= cfg_storage.min_free_gb:
            break
        if v.exists():
            rm(v)
    if removed:
        log.info("清理释放: %d 个文件", len(removed))
    return removed
