"""合成测试数据：按相机模型渲染星空 / 流星 / 飞机，生成视频和叠加快照。

没有相机时，可用它在 PC 上端到端验证 检测 → 分析 → 天体测量 → 流星雨归属 → 报表。
流星沿以辐射点为极的大圆运动，因此可以用已知真值检验测量精度。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import astro
from .calibration import CameraModel, catalog_enu, load_catalog


@dataclass
class SynthMeteor:
    t0: float                 # 相对视频开始的出现时刻 (s)
    duration: float           # 持续时间 (s)
    start_ra: float           # 起点 J2000 赤经/赤纬
    start_dec: float
    radiant_ra: float         # 运动方向：远离该辐射点
    radiant_dec: float
    speed_deg_s: float
    peak_mag: float = 0.0

    def vec_at(self, dt):
        """出现后 dt 秒时的 J2000 方向向量 (绕 radiant×start 轴旋转，即远离辐射点)。"""
        s = astro.radec_to_vec(self.start_ra, self.start_dec)
        r = astro.radec_to_vec(self.radiant_ra, self.radiant_dec)
        n = np.cross(r, s)
        n /= np.linalg.norm(n)
        a = math.radians(self.speed_deg_s * dt)
        return s * math.cos(a) + np.cross(n, s) * math.sin(a) + n * np.dot(n, s) * (1 - math.cos(a))

    def mag_at(self, dt):
        """抛物线光变：两端暗 2.5 等。"""
        u = dt / self.duration * 2 - 1
        return self.peak_mag + 2.5 * u * u


@dataclass
class SynthPlane:
    x: float                  # 起点像素 (全分辨率)
    y: float
    vx: float                 # 像素/秒
    vy: float
    flux: float = 600.0
    blink_hz: float = 1.0     # 频闪灯


def add_gaussian(img, x, y, flux, sigma=1.2):
    h, w = img.shape
    r = int(math.ceil(3 * sigma))
    xi, yi = int(round(x)), int(round(y))
    if xi < -r or yi < -r or xi >= w + r or yi >= h + r:
        return
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gy, gx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma * sigma))
    img[y0:y1, x0:x1] += flux * g / (2 * math.pi * sigma * sigma)


def render_stars(model: CameraModel, t_unix, lat, lon, zp, max_mag=6.0, sigma=1.2, extinction=True):
    img = np.zeros((model.height, model.width), np.float32)
    cat = load_catalog().brighter(max_mag)
    enu, idx = catalog_enu(cat, t_unix, lat, lon, min_alt=0.5)
    px, py = model.enu_to_pixel(enu)
    mags = cat.mag[idx].copy()
    if extinction:  # 简单大气消光 0.25 等/大气质量
        alt = np.degrees(np.arcsin(np.clip(enu[:, 2], 1e-3, 1)))
        mags += 0.25 * (1 / np.sin(np.radians(np.maximum(alt, 3))) - 1)
    for x, y, m in zip(px, py, mags):
        add_gaussian(img, x, y, 10 ** (-0.4 * (m - zp)), sigma)
    return img


def sky_background(model: CameraModel, level=30.0):
    ys, xs = np.mgrid[0:model.height, 0:model.width]
    rho = np.hypot(xs - model.x0, ys - model.y0) / model.rnorm
    bg = level * (1 + 0.5 * rho**2)  # 地平附近天光更亮
    bg[rho > 1.0] = 4.0               # 像圈外暗角
    return bg.astype(np.float32)


def render_meteor(img, model, met: SynthMeteor, t_frame, exposure, t_start, lat, lon, zp):
    """把一帧曝光期间的流星运动渲染为拖影 (子步采样)。返回真值质心或 None。"""
    a, b = max(t_frame, met.t0), min(t_frame + exposure, met.t0 + met.duration)
    if b <= a:
        return None
    pts = []
    n_sub = 8
    for k in range(n_sub):
        dt = a + (k + 0.5) * (b - a) / n_sub - met.t0
        v = met.vec_at(dt)
        ra, dec = astro.vec_to_radec(v)
        alt, az = astro.j2000_to_apparent_altaz(ra, dec, t_start + met.t0 + dt, lat, lon)
        if alt < 0:
            continue
        x, y = model.altaz_to_pixel(alt, az)
        flux = 10 ** (-0.4 * (met.mag_at(dt) - zp)) * (b - a) / exposure / n_sub
        add_gaussian(img, float(x), float(y), flux, 1.4)
        pts.append((float(x), float(y)))
    return np.mean(pts, axis=0).tolist() if pts else None


def make_video(path, model: CameraModel, t_start, duration, fps, lat, lon, meteors=(), planes=(),
               zp=9.4, noise=3.0, bg_level=30.0, seed=1, size=None):
    """写入 MJPG/AVI 合成视频，并生成同名 .json 侧车文件 (开始时间、帧率、真值)。

    size: 输出分辨率 (w, h)，默认等于 model 分辨率。
    """
    rng = np.random.default_rng(seed)
    w, h = model.width, model.height
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size or (w, h))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    bg = sky_background(model, bg_level)
    hot = rng.integers(0, [w, h], size=(30, 2))
    n_frames = int(round(duration * fps))
    exposure = 1.0 / fps
    truth = {i: [] for i in range(len(meteors))}
    stars, stars_t = None, -1e9
    for fi in range(n_frames):
        t_rel = fi / fps
        if t_rel - stars_t >= 5.0:  # 星空每 5 秒更新一次 (周日运动)
            stars, stars_t = render_stars(model, t_start + t_rel, lat, lon, zp), t_rel
        img = bg + stars
        for i, met in enumerate(meteors):
            c = render_meteor(img, model, met, t_rel, exposure, t_start, lat, lon, zp)
            if c is not None:
                truth[i].append({"frame": fi, "t": t_start + t_rel + exposure / 2, "x": c[0], "y": c[1]})
        for p in planes:
            on = (t_rel * p.blink_hz) % 1.0 < 0.5
            add_gaussian(img, p.x + p.vx * t_rel, p.y + p.vy * t_rel, p.flux * (1.0 if on else 0.3), 1.3)
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
        img[hot[:, 1], hot[:, 0]] = 200
        frame = np.clip(img, 0, 255).astype(np.uint8)
        if size and size != (w, h):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    side = {"start_unix": t_start, "fps": fps, "frames": n_frames,
            "meteors": [asdict(m) for m in meteors], "truth": truth}
    with open(str(path).rsplit(".", 1)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1)
    return side


def make_snapshot(model: CameraModel, t_unix, lat, lon, zp=9.4, noise=3.0, n_stack=32, bg_level=30.0, seed=2):
    """模拟采集服务的叠加快照 (n 帧平均)。"""
    rng = np.random.default_rng(seed)
    img = sky_background(model, bg_level) + render_stars(model, t_unix, lat, lon, zp)
    img += rng.normal(0, noise / math.sqrt(n_stack), img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.float32)
