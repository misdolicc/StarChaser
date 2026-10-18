"""离线事件分析：对预录片段做精确的流星提取、测量与判别。

1. 叠加 (流式，内存恒定): maxpixel / maxframe (每像素最亮帧号) / avepixel / stdpixel
   (平均与标准差剔除最大值，避免流星本身抬高背景；思路同 GMN/RMS 的 FF 文件)
2. 候选像素: maxpixel - avepixel > k·σ
3. 在 (x, y, 帧号) 空间做 RANSAC：流星 = 随时间匀速移动的线段，同时满足空间共线与时间单调，
   天然排除闪烁星、热像素、噪声；可逐条剥离提取多颗流星
4. 每帧质心 → 平滑拟合；第二遍解码做孔径测光并截取关键帧
5. 天体测量 (measure): 像素 → 地平 → J2000 赤道；角长度/角速度/星等/流星雨归属
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import astro, showers
from .calibration import CameraModel

log = logging.getLogger(__name__)


@dataclass
class Stacks:
    maxpix: np.ndarray      # uint8 灰度
    maxframe: np.ndarray    # uint16
    ave: np.ndarray         # float32
    std: np.ndarray         # float32
    maxcolor: np.ndarray    # uint8 BGR
    n: int
    roi: tuple              # (x, y, w, h)，全帧坐标中的裁剪区域


@dataclass
class TrackFit:
    frames: np.ndarray       # 帧号
    cx: np.ndarray           # 原始每帧质心 (全帧坐标)
    cy: np.ndarray
    weight: np.ndarray       # 每帧信号总和
    px: np.ndarray           # 拟合后的平滑位置 (每帧)
    py: np.ndarray
    rms: float
    coverage: float
    n_points: int
    flux: np.ndarray = field(default_factory=lambda: np.zeros(0))
    saturated: bool = False

    @property
    def length_px(self):
        return float(math.hypot(self.px[-1] - self.px[0], self.py[-1] - self.py[0]))

    @property
    def speed_px(self):
        span = self.frames[-1] - self.frames[0]
        return self.length_px / span if span > 0 else 0.0


def roi_from_mask(mask, pad=4):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(mask.shape[1], xs.max() + pad + 1), min(mask.shape[0], ys.max() + pad + 1)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def iter_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频 {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def compute_stacks(path, roi=None) -> Stacks:
    maxg = maxf = s = s2 = maxc = None
    n = 0
    for frame in iter_frames(path):
        if roi is None:
            roi = (0, 0, frame.shape[1], frame.shape[0])
        x, y, w, h = roi
        fc = frame[y:y + h, x:x + w]
        g = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY) if fc.ndim == 3 else fc
        if maxg is None:
            maxg = g.copy()
            maxf = np.zeros(g.shape, np.uint16)
            s = g.astype(np.float32)
            s2 = s * s
            maxc = fc.copy() if fc.ndim == 3 else cv2.cvtColor(fc, cv2.COLOR_GRAY2BGR)
        else:
            upd = g > maxg
            maxg[upd] = g[upd]
            maxf[upd] = n
            gf = g.astype(np.float32)
            s += gf
            s2 += gf * gf
            np.maximum(maxc, fc if fc.ndim == 3 else cv2.cvtColor(fc, cv2.COLOR_GRAY2BGR), out=maxc)
        n += 1
    if n < 3:
        raise ValueError(f"视频帧数过少 ({n})")
    mx = maxg.astype(np.float32)
    ave = (s - mx) / (n - 1)
    var = (s2 - mx * mx) / (n - 1) - ave * ave
    return Stacks(maxg, maxf, ave, np.sqrt(np.clip(var, 0, None)), maxc, n, roi)


# ---------------------------------------------------------------- RANSAC 航迹
def _predict(coef_x, coef_y, f):
    px, py = np.polyval(coef_x, f), np.polyval(coef_y, f)
    vx, vy = np.polyval(np.polyder(coef_x), f), np.polyval(np.polyder(coef_y), f)
    return px, py, vx, vy


def _inliers(x, y, f, coef_x, coef_y, tol_cross):
    px, py, vx, vy = _predict(coef_x, coef_y, f)
    sp = np.hypot(vx, vy) + 1e-9
    dx, dy = x - px, y - py
    along = (dx * vx + dy * vy) / sp
    cross = (-dx * vy + dy * vx) / sp
    return (np.abs(cross) <= tol_cross) & (np.abs(along) <= 0.75 * sp + 3.0)


def ransac_tracks(x, y, f, w, min_frames=4, max_tracks=3, max_speed_px=80.0, iters=500, seed=0):
    """返回 [(inlier_indices, coef_x, coef_y)]，坐标单位与输入一致。"""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(x))
    out = []
    for _ in range(max_tracks):
        if len(remaining) < 2 * min_frames:
            break
        if len(remaining) > 6000:  # 假设评估用强度最高的子集
            sub = remaining[np.argsort(-w[remaining])[:6000]]
        else:
            sub = remaining
        xs, ys, fs, ws = x[sub], y[sub], f[sub], w[sub]
        p = ws / ws.sum()
        best, best_score = None, 0
        for _ in range(iters):
            i = rng.choice(len(sub), p=p)
            d = np.hypot(xs - xs[i], ys - ys[i])
            near = np.nonzero((d > 1.5) & (d < 120) & (fs != fs[i]))[0]
            if len(near) == 0:
                continue
            j = near[rng.integers(len(near))]
            df = fs[j] - fs[i]
            vx, vy = (xs[j] - xs[i]) / df, (ys[j] - ys[i]) / df
            sp = math.hypot(vx, vy)
            if sp < 0.2 or sp > max_speed_px:
                continue
            cx, cy = np.array([vx, xs[i] - vx * fs[i]]), np.array([vy, ys[i] - vy * fs[i]])
            inl = _inliers(xs, ys, fs, cx, cy, 3.0)
            nf = len(np.unique(fs[inl]))
            score = int(inl.sum()) * min(nf, 30)
            if nf >= min_frames and score > best_score:
                best, best_score = (cx, cy), score
        if best is None:
            break
        cx, cy = best
        xr, yr, fr, wr = x[remaining], y[remaining], f[remaining], w[remaining]
        inl = _inliers(xr, yr, fr, cx, cy, 3.0)
        for it in range(3):  # 精修：加权最小二乘，帧数多时允许二次项 (鱼眼弯曲 + 减速)
            if inl.sum() < 3:
                break
            deg = 2 if len(np.unique(fr[inl])) >= 10 else 1
            sw = np.sqrt(np.clip(wr[inl], 1e-3, None))
            cx = np.polyfit(fr[inl], xr[inl], deg, w=sw)
            cy = np.polyfit(fr[inl], yr[inl], deg, w=sw)
            inl = _inliers(xr, yr, fr, cx, cy, 6.0 if it else 4.0)
        if len(np.unique(fr[inl])) < min_frames:
            break
        out.append((remaining[inl], cx, cy))
        remaining = remaining[~inl]
    return out


def centroid_track(x, y, f, w, idx, cx, cy) -> TrackFit:
    fi = f[idx].astype(int)
    frames = np.unique(fi)
    cxs, cys, ws, keep = [], [], [], []
    for fr in frames:
        k = idx[fi == fr]
        ww = w[k]
        if len(k) < 2 or ww.sum() <= 0:
            continue
        cxs.append(float((x[k] * ww).sum() / ww.sum()))
        cys.append(float((y[k] * ww).sum() / ww.sum()))
        ws.append(float(ww.sum()))
        keep.append(fr)
    frames = np.array(keep)
    cxs, cys, ws = np.array(cxs), np.array(cys), np.array(ws)
    if len(frames) < 2:
        return TrackFit(frames, cxs, cys, ws, cxs, cys, 99.0, 0.0, len(idx))
    deg = 2 if len(frames) >= 10 else 1
    sw = np.sqrt(ws)
    fx = np.polyfit(frames, cxs, deg, w=sw)
    fy = np.polyfit(frames, cys, deg, w=sw)
    px, py = np.polyval(fx, frames), np.polyval(fy, frames)
    rms = float(np.sqrt(np.mean((px - cxs) ** 2 + (py - cys) ** 2)))
    coverage = len(frames) / (frames[-1] - frames[0] + 1)
    return TrackFit(frames, cxs, cys, ws, px, py, rms, coverage, len(idx))


def find_tracks(st: Stacks, mask_roi, acfg):
    """在叠加图中提取航迹。返回 ([TrackFit], 诊断信息)。坐标为全帧坐标。"""
    diff = st.maxpix.astype(np.float32) - st.ave
    std = np.minimum(st.std, cv2.medianBlur(st.std, 5))  # 抑制流星自身对 σ 的抬高
    thr = np.maximum(acfg.k_sigma * np.maximum(std, 1.0), acfg.min_delta)
    cand = ((diff > thr) & (mask_roi > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= 3
    big[0] = False
    ys, xs = np.nonzero(big[lab])
    info = {"candidates": int(len(xs)), "mask_px": int((mask_roi > 0).sum())}
    if len(xs) < acfg.min_frames * 2:
        return [], info
    if len(xs) > 0.05 * info["mask_px"]:
        info["noisy"] = True  # 云/闪电/增益变化导致大面积变化
        keep = np.argsort(-diff[ys, xs])[:20000]
        ys, xs = ys[keep], xs[keep]
    f = st.maxframe[ys, xs].astype(np.float64)
    w = diff[ys, xs].astype(np.float64)
    rx, ry = st.roi[0], st.roi[1]
    x, y = xs + float(rx), ys + float(ry)
    tracks = []
    for idx, cx, cy in ransac_tracks(x, y, f, w, acfg.min_frames, acfg.max_tracks):
        tf = centroid_track(x, y, f, w, idx, cx, cy)
        if len(tf.frames) >= 2:
            tracks.append(tf)
    return tracks, info


# ---------------------------------------------------------------- 第二遍：测光 + 关键帧
def photometry_pass(path, tracks, st: Stacks, crop=64, max_tiles=40):
    """孔径测光、饱和检测；收集起点/峰值/终点全帧与每帧小图。"""
    x0, y0, w, h = st.roi
    plans = []
    for t in tracks:
        f0, f1 = int(t.frames[0]), int(t.frames[-1])
        dfx = np.polyfit(t.frames, t.px, min(2, len(t.frames) - 1))
        dfy = np.polyfit(t.frames, t.py, min(2, len(t.frames) - 1))
        tiles = list(range(f0, f1 + 1))
        if len(tiles) > max_tiles:
            tiles = list(np.linspace(f0, f1, max_tiles).round().astype(int))
        plans.append({"f0": f0, "f1": f1, "fx": dfx, "fy": dfy, "flux": {}, "sat": False, "tiles": set(tiles),
                      "crops": [], "start": None, "end": None, "peak": None, "peak_flux": -1e18,
                      "radius": 4.0 + 0.6 * t.speed_px})
    for i, frame in enumerate(iter_frames(path)):
        fc = frame[y0:y0 + h, x0:x0 + w]
        g = None
        for p in plans:
            if not (p["f0"] <= i <= p["f1"]):
                continue
            if g is None:
                g = (cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY) if fc.ndim == 3 else fc).astype(np.float32)
            cx, cy = float(np.polyval(p["fx"], i)) - x0, float(np.polyval(p["fy"], i)) - y0
            r = p["radius"]
            a, b = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
            c, d = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
            if a >= b or c >= d:
                continue
            yy, xx = np.ogrid[a:b, c:d]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            flux = float(((g[a:b, c:d] - st.ave[a:b, c:d]) * disk).sum())
            p["flux"][i] = flux
            if g[a:b, c:d][disk].max(initial=0) >= 250:
                p["sat"] = True
            if i == p["f0"]:
                p["start"] = fc.copy()
            if i == p["f1"]:
                p["end"] = fc.copy()
            if flux > p["peak_flux"]:
                p["peak_flux"], p["peak"], p["peak_i"] = flux, fc.copy(), i
            if i in p["tiles"]:
                p["crops"].append((i, _crop(fc, cx, cy, crop)))
    for t, p in zip(tracks, plans):
        t.flux = np.array([p["flux"].get(int(fr), 0.0) for fr in t.frames])
        t.saturated = p["sat"]
    return plans


def _crop(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------- 天体测量与判别
def frame_times(clip: dict, n: int):
    ts = clip.get("frame_ts") or []
    fps = float(clip.get("fps") or 25.0)
    start = clip.get("start_ts") or (ts[0] if ts else 0.0)
    return np.array([ts[i] if i < len(ts) else start + i / fps for i in range(n)])


def measure(track: dict, model: CameraModel | None, site: dict, gain: float | None, fps: float, calibrated: bool):
    """由每帧像素位置计算天球量。track: {frames, t, x, y, flux}。可在标定更新后重新调用。"""
    t = np.asarray(track["t"], float)
    x, y = np.asarray(track["x"], float), np.asarray(track["y"], float)
    flux = np.asarray(track.get("flux") or np.zeros(len(t)), float)
    out = {"t_start": float(t[0]), "t_end": float(t[-1]),
           "duration": (int(track["frames"][-1]) - int(track["frames"][0]) + 1) / fps,
        "x0": float(x[0]), "y0": float(y[0]), "x1": float(x[-1]), "y1": float(y[-1]),
        "length_px": float(math.hypot(x[-1] - x[0], y[-1] - y[0])), "n_frames": int(len(t)),
        "calibrated": int(bool(calibrated))}
    ip = int(np.argmax(flux)) if len(flux) and flux.max() > 0 else len(t) // 2
    out["t_peak"] = float(t[ip])
    if model is None:
        return out
    alt, az = model.pixel_to_altaz(x, y)
    v = astro.altaz_to_enu(alt, az)
    length = float(astro.vec_angle(v[0], v[-1]))
    dt = float(t[-1] - t[0])
    # 未标定时模型只是按镜头参数的近似：角长度/角速度可用于判别，方向 (方位角) 不可信，不输出
    out.update(length_deg=length, speed_deg_s=length / dt if dt > 0 else None)
    if calibrated:
        out.update(alt0=float(alt[0]), az0=float(az[0]), alt1=float(alt[-1]), az1=float(az[-1]))
    if model.zp is not None and gain and flux.max(initial=0) > 0:
        zp = model.zp + 2.5 * math.log10(gain / (model.zp_gain or gain))
        out["peak_mag"] = round(zp - 2.5 * math.log10(flux[ip]), 2)
        out["mags"] = [round(zp - 2.5 * math.log10(f), 2) if f > 0 else None for f in flux]
    if calibrated:
        ra, dec = astro.apparent_altaz_to_j2000(alt, az, t, site["lat"], site["lon"])
        out.update(ra0=float(ra[0]), dec0=float(dec[0]), ra1=float(ra[-1]), dec1=float(dec[-1]))
        code, dist = showers.associate(ra[0], dec[0], ra[-1], dec[-1], float(t.mean()), site["lat"], site["lon"])
        out.update(shower=code, shower_dist=dist)
        out["track_radec"] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(ra, dec)]
    return out


def classify(m: dict, tf: TrackFit, acfg, fps):
    """返回拒绝原因列表 (空 = 判定为流星)。"""
    reasons = []
    if len(tf.frames) < acfg.min_frames:
        reasons.append(f"帧数不足({len(tf.frames)})")
    if tf.length_px < acfg.min_length_px:
        reasons.append(f"轨迹过短({tf.length_px:.0f}px)")
    if tf.rms > acfg.max_linear_rms_px:
        reasons.append(f"非直线(rms {tf.rms:.1f}px)")
    if tf.coverage < 0.5:
        reasons.append(f"断续({tf.coverage:.0%})")
    if m["duration"] > acfg.max_duration_s:
        reasons.append(f"持续过长({m['duration']:.1f}s)")
    sp = m.get("speed_deg_s")
    if sp is not None:
        if sp < acfg.min_speed_deg_s:
            reasons.append(f"太慢({sp:.1f}°/s)")
        if sp > acfg.max_speed_deg_s:
            reasons.append(f"太快({sp:.0f}°/s)")
    return reasons


def analyze_clip(video, meta: dict, cfg, model: CameraModel | None, calibrated: bool, mask_full):
    """完整分析一个事件片段。返回 dict(tracks=[...], stacks, plans, info)。"""
    acfg = cfg.analysis
    t0 = time.time()
    roi = roi_from_mask(mask_full)
    st = compute_stacks(video, roi)
    x, y, w, h = st.roi
    tracks, info = find_tracks(st, mask_full[y:y + h, x:x + w], acfg)
    plans = photometry_pass(video, tracks, st) if tracks else []
    fps = float(meta["clip"].get("fps") or cfg.camera.fps)
    times = frame_times(meta["clip"], st.n)
    site = meta.get("site") or {"lat": cfg.site.latitude, "lon": cfg.site.longitude}
    gain = (meta.get("camera") or {}).get("gain")
    results = []
    for tf, plan in zip(tracks, plans):
        tr = {"frames": tf.frames.tolist(), "t": [float(times[min(int(fr), st.n - 1)]) for fr in tf.frames],
              "x": tf.px.tolist(), "y": tf.py.tolist(), "cx": tf.cx.tolist(), "cy": tf.cy.tolist(),
              "flux": tf.flux.tolist()}
        m = measure(tr, model, site, gain, fps, calibrated)
        m["saturated"] = int(tf.saturated)
        reasons = classify(m, tf, acfg, fps)
        results.append({"track": tr, "measure": m, "fit": {"rms_px": tf.rms, "coverage": tf.coverage,
                        "points": tf.n_points}, "reasons": reasons, "plan": plan})
    info.update(frames=st.n, seconds=round(time.time() - t0, 2), tracks_found=len(tracks))
    return {"results": results, "stacks": st, "info": info}
