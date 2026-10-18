"""实时流星检测 (运行在低分辨率 lores 流上，树莓派 Zero 2W 单核可跑 25fps)。

每帧:
    1. diff = 帧 - 背景 (EMA)，阈值 = max(k·σ, min_delta)，σ 由逐像素平均绝对偏差估计
       —— 闪烁的亮星、热像素噪声大，阈值自动升高
    2. 背景只在未检出处正常更新，避免流星污染背景
    3. 检出像素过多 → 全局变化 (闪电、车灯、云、增益调整)，丢弃本帧
    4. 连通域 → 斑点 (通量加权质心)
    5. 与航迹做最近邻关联 (按速度外推)，满足 帧数/位移/速度/直线度 后确认为候选流星
确认后由录像器保存预录视频，精确判别留给离线分析 (analysis.py)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


def _cluster_small(xs, ys, link=2.0):
    """少量像素的连通聚类 (切比雪夫距离 ≤ 2，等价于 3×3 膨胀后的 8 连通)。返回 (标签 1..n-1, n)。"""
    k = len(xs)
    near = (np.abs(xs[:, None] - xs[None, :]) <= link) & (np.abs(ys[:, None] - ys[None, :]) <= link)
    lab = np.zeros(k, np.int64)
    n = 1
    for i in range(k):
        if lab[i]:
            continue
        lab[i] = n
        stack = [i]
        while stack:
            j = stack.pop()
            for m in np.nonzero(near[j] & (lab == 0))[0]:
                lab[m] = n
                stack.append(m)
        n += 1
    return lab, n


@dataclass
class TrackPoint:
    fi: int
    ts: float
    x: float
    y: float
    flux: float
    area: int


@dataclass
class Track:
    id: int
    points: list = field(default_factory=list)
    confirmed: bool = False
    too_long: bool = False

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def first(self) -> TrackPoint:
        return self.points[0]

    def velocity(self):
        """最近若干点的线性拟合速度 (像素/帧)。"""
        pts = self.points[-6:]
        if len(pts) < 2:
            return 0.0, 0.0
        f = np.array([p.fi for p in pts], float)
        f -= f.mean()
        den = float((f * f).sum()) or 1.0
        vx = float((f * np.array([p.x for p in pts])).sum() / den)
        vy = float((f * np.array([p.y for p in pts])).sum() / den)
        return vx, vy

    def predict(self, fi):
        vx, vy = self.velocity()
        dt = fi - self.last.fi
        return self.last.x + vx * dt, self.last.y + vy * dt

    def gate(self, fi, gate_px):
        if len(self.points) < 2:
            return gate_px
        vx, vy = self.velocity()
        return 2.5 + 0.5 * math.hypot(vx, vy) * (fi - self.last.fi)

    def stats(self):
        xs = np.array([p.x for p in self.points])
        ys = np.array([p.y for p in self.points])
        span = self.last.fi - self.first.fi
        disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        rms, mono = 0.0, 1.0
        if len(xs) >= 3:
            c = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
            _, s, vt = np.linalg.svd(c, full_matrices=False)
            rms = float(s[1] / math.sqrt(len(xs)))   # 垂直于主方向的残差
            proj = c @ vt[0]
            steps = np.diff(proj)
            if (proj[-1] - proj[0]) < 0:
                steps = -steps
            mono = float(np.mean(steps >= -0.5))
        speed = disp / span if span > 0 else 0.0
        return {"n": len(xs), "span": span, "disp": disp, "speed": speed, "rms": rms, "mono": mono}

    def to_dict(self):
        return {"id": self.id, "too_long": self.too_long,
                "points": [[p.fi, round(p.ts, 4), round(p.x, 2), round(p.y, 2), round(p.flux, 1), p.area]
                           for p in self.points]}


@dataclass
class DetectResult:
    new_confirmed: list
    closed_confirmed: list
    active_confirmed: bool
    global_event: bool = False
    warming: bool = False
    n_blobs: int = 0


class RealtimeDetector:
    def __init__(self, cfg, shape, mask=None, fps=25.0):
        self.c = cfg
        self.h, self.w = shape
        self.fps = fps
        self.mask = (mask > 0) if mask is not None else np.ones(shape, bool)
        self.mask_area = max(1, int(self.mask.sum()))
        self.kernel = np.ones((3, 3), np.uint8)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.reset()

    def reset(self):
        """增益改变或重新开始时调用：重新学习背景。"""
        self.bg = None
        self.noise = None
        self.warm = int(self.c.warmup_s * self.fps)
        self.tracks = [t for t in self.tracks if t.confirmed]
        self._fcount = 0

    def process(self, gray: np.ndarray, fi: int, ts: float) -> DetectResult:
        c = self.c
        f = gray.astype(np.float32)
        if self.bg is None:
            self.bg = f.copy()
            self.noise = np.full_like(f, 2.0)
            self._update_thr()
            return DetectResult([], [], False, warming=True)

        diff = cv2.subtract(f, self.bg)
        det_u8 = cv2.compare(diff, self.thr, cv2.CMP_GT)  # 255 = 检出；掩膜外阈值为无穷大
        n_det = cv2.countNonZero(det_u8)
        nondet_u8 = cv2.bitwise_not(det_u8) if n_det else None

        # 背景只在未检出处更新 (避免流星污染)；每 16 帧全图慢速更新一次，让持久变化 (新亮起的灯) 融入背景
        self._fcount += 1
        cv2.accumulateWeighted(f, self.bg, c.bg_alpha, mask=nondet_u8)
        if n_det and self._fcount % 16 == 0:
            cv2.accumulateWeighted(f, self.bg, c.bg_alpha * 0.5)
        if self._fcount % 4 == 0:  # 噪声图每 4 帧更新一次以节省 CPU
            ad = cv2.min(np.abs(diff), self.noise * 4 + 2)
            cv2.accumulateWeighted(ad, self.noise, c.noise_alpha * 4, mask=nondet_u8)
            self._update_thr()

        if self.warm > 0:
            self.warm -= 1
            return DetectResult([], [], False, warming=True)

        if n_det > c.global_change_fraction * self.mask_area:
            return self._global(fi)

        blobs = []
        if n_det:
            pts = cv2.findNonZero(det_u8).reshape(-1, 2)
            xs, ys = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
            wgt = diff[pts[:, 1], pts[:, 0]].astype(np.float64)
            if n_det <= 48:  # 常见情况：只有零星几个像素，直接聚类，省去整幅图的连通域分析
                lab, n = _cluster_small(xs, ys)
            else:
                n, labels = cv2.connectedComponents(cv2.dilate(det_u8, self.kernel), connectivity=8)
                lab = labels[pts[:, 1], pts[:, 0]]
            if n - 1 > c.max_blobs:
                return self._global(fi)
            sw = np.bincount(lab, weights=wgt, minlength=n)
            sx = np.bincount(lab, weights=wgt * xs, minlength=n)
            sy = np.bincount(lab, weights=wgt * ys, minlength=n)
            area = np.bincount(lab, minlength=n)
            for k in range(1, n):
                if area[k] >= c.min_area and sw[k] > 0:
                    blobs.append((sx[k] / sw[k], sy[k] / sw[k], float(sw[k]), int(area[k])))
        return self._track(blobs, fi, ts)

    def _update_thr(self):
        """阈值 = max(k·σ, min_delta)，σ = 1.2533 × 平均绝对偏差；掩膜外设为无穷大。"""
        c = self.c
        self.thr = np.maximum(self.noise * (c.k_sigma * 1.2533), np.float32(c.min_delta))
        self.thr[~self.mask] = np.inf

    def _global(self, fi):
        self.tracks = [t for t in self.tracks if t.confirmed]
        closed = self._close_stale(fi, force_unconfirmed=True)
        return DetectResult([], closed, any(t.confirmed for t in self.tracks), global_event=True)

    def _close_stale(self, fi, force_unconfirmed=False):
        closed, keep = [], []
        max_len = self.c.max_track_s * self.fps
        for t in self.tracks:
            stale = fi - t.last.fi > self.c.max_gap_frames
            if t.last.fi - t.first.fi > max_len:
                t.too_long, stale = True, True
            if stale or (force_unconfirmed and not t.confirmed):
                if t.confirmed:
                    closed.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return closed

    def _track(self, blobs, fi, ts):
        c = self.c
        blobs.sort(key=lambda b: -b[2])
        assigned = set()
        new_confirmed = []
        # 长航迹优先关联
        for t in sorted(self.tracks, key=lambda t: -len(t.points)):
            if not blobs:
                break
            px, py = t.predict(fi)
            g = t.gate(fi, c.gate_px)
            best, bd = None, g
            for i, b in enumerate(blobs):
                if i in assigned:
                    continue
                d = math.hypot(b[0] - px, b[1] - py)
                if d < bd:
                    best, bd = i, d
            if best is not None:
                assigned.add(best)
                b = blobs[best]
                t.points.append(TrackPoint(fi, ts, b[0], b[1], b[2], b[3]))
                if not t.confirmed and self._is_meteor_like(t):
                    t.confirmed = True
                    new_confirmed.append(t)
        for i, b in enumerate(blobs):
            if i not in assigned:
                self.tracks.append(Track(self.next_id, [TrackPoint(fi, ts, b[0], b[1], b[2], b[3])]))
                self.next_id += 1
        if len(self.tracks) > 80:  # 限制航迹数量：丢弃最旧的未确认航迹
            unconf = sorted((t for t in self.tracks if not t.confirmed), key=lambda t: t.last.fi)
            drop = {id(t) for t in unconf[: len(self.tracks) - 80]}
            self.tracks = [t for t in self.tracks if id(t) not in drop]
        closed = self._close_stale(fi)
        return DetectResult(new_confirmed, closed, any(t.confirmed for t in self.tracks), n_blobs=len(blobs))

    def _is_meteor_like(self, t: Track) -> bool:
        c = self.c
        s = t.stats()
        if s["span"] <= 0 or s["mono"] < 0.75 or s["rms"] > c.linear_rms_px:
            return False
        if s["speed"] < c.min_speed_px:
            return False
        if s["n"] >= c.min_frames and s["disp"] >= c.min_displacement_px:
            return True
        # 快速流星可能只出现 3 帧
        return s["n"] >= 3 and s["disp"] >= 3 * c.min_displacement_px
