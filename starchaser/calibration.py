"""全天鱼眼镜头的天体测量标定。

镜头模型 (径向多项式 + 光轴倾斜，参考 Borovička 1995 全天相机模型的简化形式):
    ρ = |p - c| / rnorm,  θ = k1·ρ + k2·ρ² + k3·ρ³        (θ: 离光轴的角距，弧度)
    ψ = atan2(dy, dx)  (镜像时取负)
    v_cam = (sinθ·cosψ, sinθ·sinψ, cosθ)
    v_enu = Rx(tilt_x)·Ry(tilt_y)·Rz(roll) · v_cam          (东-北-天 坐标系)

标定流程 (无需人工认星):
    1. 在夜间叠加快照中检测星点
    2. 用星表计算当时的视位置 (岁差 + 折射)
    3. 粗搜索：遍历 roll / 镜像 / 焦距尺度，最大化匹配星数
    4. 逐步收紧匹配容差，最小二乘拟合全部参数
    5. 用匹配星拟合测光零点
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import astro

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
PARAM_NAMES = ("x0", "y0", "k1", "k2", "k3", "tilt_x", "tilt_y", "roll")


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass
class CameraModel:
    width: int
    height: int
    x0: float
    y0: float
    rnorm: float
    k1: float
    k2: float = 0.0
    k3: float = 0.0
    tilt_x: float = 0.0   # 度
    tilt_y: float = 0.0   # 度
    roll: float = 0.0     # 度
    mirror: bool = False
    zp: float | None = None   # 测光零点: m = zp - 2.5·log10(单帧通量)，对应增益 zp_gain
    zp_gain: float = 1.0
    rms_arcmin: float | None = None
    n_stars: int = 0
    fitted_at: str = ""
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------- 构造 / 序列化
    @classmethod
    def initial(cls, width, height, fov_deg=185.0, circle_radius=0, north_angle=-1.0, mirror=False):
        r = circle_radius or min(width, height) / 2.0
        roll = 0.0
        if north_angle is not None and north_angle >= 0:
            # 北方向在图像中的极角 φ_N (y 向下，顺时针为正): 上方 = -90°
            phi_n = -90.0 + north_angle
            roll = (phi_n + 90.0) if mirror else (90.0 - phi_n)
        return cls(width, height, width / 2.0, height / 2.0, r, math.radians(fov_deg / 2.0),
                   roll=roll % 360.0, mirror=mirror)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_params(self, names, values):
        d = self.to_dict()
        d.update({n: float(v) for n, v in zip(names, values)})
        return CameraModel.from_dict(d)

    def scaled(self, width, height):
        """换算到另一分辨率 (同视场的缩放流，例如 lores)。"""
        sx, sy = width / self.width, height / self.height
        d = self.to_dict()
        d.update(width=width, height=height, x0=self.x0 * sx, y0=self.y0 * sy, rnorm=self.rnorm * (sx + sy) / 2)
        return CameraModel.from_dict(d)

    # ---------------------------------------------------------- 投影
    def _rot(self):
        return _rx(math.radians(self.tilt_x)) @ _ry(math.radians(self.tilt_y)) @ _rz(math.radians(self.roll))

    def pixel_to_enu(self, x, y):
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        rho = np.hypot(dx, dy) / self.rnorm
        psi = np.arctan2(dy, dx)
        if self.mirror:
            psi = -psi
        theta = self.k1 * rho + self.k2 * rho**2 + self.k3 * rho**3
        st = np.sin(theta)
        vc = np.stack([st * np.cos(psi), st * np.sin(psi), np.cos(theta)], axis=-1)
        return vc @ self._rot().T

    def enu_to_pixel(self, v):
        vc = np.asarray(v, dtype=float) @ self._rot()
        theta = np.arccos(np.clip(vc[..., 2], -1.0, 1.0))
        psi = np.arctan2(vc[..., 1], vc[..., 0])
        if self.mirror:
            psi = -psi
        rho = theta / self.k1
        for _ in range(8):  # 牛顿迭代反解 θ(ρ)
            f = self.k1 * rho + self.k2 * rho**2 + self.k3 * rho**3 - theta
            fp = self.k1 + 2 * self.k2 * rho + 3 * self.k3 * rho**2
            rho = rho - f / np.where(np.abs(fp) < 1e-9, 1e-9, fp)
        r = rho * self.rnorm
        return self.x0 + r * np.cos(psi), self.y0 + r * np.sin(psi)

    def pixel_to_altaz(self, x, y):
        return astro.enu_to_altaz(self.pixel_to_enu(x, y))

    def altaz_to_pixel(self, alt, az):
        return self.enu_to_pixel(astro.altaz_to_enu(alt, az))

    def pixel_to_radec(self, x, y, t_unix, lat, lon):
        alt, az = self.pixel_to_altaz(x, y)
        return astro.apparent_altaz_to_j2000(alt, az, t_unix, lat, lon)

    def sky_mask(self, min_alt=0.0, width=None, height=None, step=4):
        """高度角 > min_alt 且位于像圈内的有效区域掩膜 (uint8 0/255)。降采样计算以节省内存。"""
        w, h = width or self.width, height or self.height
        m = self.scaled(max(1, w // step), max(1, h // step))
        ys, xs = np.mgrid[0:m.height, 0:m.width] + 0.5
        alt, _ = m.pixel_to_altaz(xs, ys)
        rho = np.hypot(xs - m.x0, ys - m.y0) / m.rnorm
        small = ((alt > min_alt) & (rho <= 1.0)).astype(np.uint8) * 255
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def circle_mask(width, height, radius=0, cx=None, cy=None):
    r = radius or min(width, height) / 2.0
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    m = np.zeros((height, width), np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), int(r), 255, -1)
    return m


# ---------------------------------------------------------------- 星表
@dataclass
class Catalog:
    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray
    name: list

    def brighter(self, max_mag):
        k = self.mag <= max_mag
        return Catalog(self.ra[k], self.dec[k], self.mag[k], [n for n, ok in zip(self.name, k) if ok])


_CATALOG_CACHE: Catalog | None = None


def load_catalog(path=None) -> Catalog:
    global _CATALOG_CACHE
    if path is None and _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    p = Path(path) if path else DATA_DIR / "stars.csv"
    ra, dec, mag, name = [], [], [], []
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ra.append(float(row["ra_deg"]))
            dec.append(float(row["dec_deg"]))
            mag.append(float(row["vmag"]))
            name.append(row["name"])
    cat = Catalog(np.array(ra), np.array(dec), np.array(mag), name)
    if path is None:
        _CATALOG_CACHE = cat
    return cat


def catalog_enu(cat: Catalog, t_unix, lat, lon, min_alt=0.0):
    """星表在 t 时刻的视位置 (ENU 单位向量)。返回 (enu, 索引)。"""
    alt, az = astro.j2000_to_apparent_altaz(cat.ra, cat.dec, t_unix, lat, lon)
    keep = np.nonzero(alt > min_alt)[0]
    return astro.altaz_to_enu(alt[keep], az[keep]), keep


# ---------------------------------------------------------------- 星点检测
def detect_stars(img, mask=None, k_sigma=5.0, max_stars=600, saturation=250.0):
    """在 (叠加) 图像中检测星点。返回 dict(x, y, flux, peak) 数组，按通量降序。"""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bg = cv2.GaussianBlur(cv2.morphologyEx(img, cv2.MORPH_OPEN, kern), (0, 0), 6)
    res = img - bg
    valid = np.ones(img.shape, bool) if mask is None else mask > 0
    # 边缘 4 像素内的星点无法完整测量
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    vals = res[valid]
    if vals.size < 100:
        return {"x": np.zeros(0), "y": np.zeros(0), "flux": np.zeros(0), "peak": np.zeros(0)}
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-3
    local_max = res >= cv2.dilate(res, np.ones((5, 5), np.uint8))
    cand = local_max & valid & (res > med + k_sigma * sigma)
    ys, xs = np.nonzero(cand)
    order = np.argsort(-res[ys, xs])[: max_stars * 3]
    ys, xs = ys[order], xs[order]
    h, w = img.shape
    out_x, out_y, out_f, out_p = [], [], [], []
    dy_g, dx_g = np.mgrid[-3:4, -3:4]
    for y, x in zip(ys, xs):
        if y < 3 or x < 3 or y >= h - 3 or x >= w - 3:
            continue
        win = res[y - 3:y + 4, x - 3:x + 4]
        pk = res[y, x]
        # 热像素: 单个孤立亮点，周围几乎没有信号
        neigh = (win[2, 3] + win[4, 3] + win[3, 2] + win[3, 4]) / 4.0
        if neigh < 0.15 * pk:
            continue
        wpos = np.clip(win - med, 0, None)
        s = wpos.sum()
        if s <= 0:
            continue
        out_x.append(x + float((wpos * dx_g).sum() / s))
        out_y.append(y + float((wpos * dy_g).sum() / s))
        out_f.append(float(win.sum() - med * win.size))
        out_p.append(float(img[y, x]))
        if len(out_x) >= max_stars:
            break
    order = np.argsort(-np.array(out_f)) if out_f else []
    return {key: np.array(arr)[order] if len(arr) else np.zeros(0)
            for key, arr in (("x", out_x), ("y", out_y), ("flux", out_f), ("peak", out_p))}


# ---------------------------------------------------------------- 匹配与拟合
def match_nearest(v_det, v_cat, tol_deg):
    """互为最近邻且角距 < tol 的匹配对。返回 (det_idx, cat_idx)。"""
    if len(v_det) == 0 or len(v_cat) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    dots = v_det @ v_cat.T
    j = np.argmax(dots, axis=1)
    i_back = np.argmax(dots, axis=0)
    i = np.arange(len(v_det))
    ok = (dots[i, j] >= math.cos(math.radians(tol_deg))) & (i_back[j] == i)
    return i[ok], j[ok]


def _fit(model, x, y, v_target, names):
    from scipy.optimize import least_squares

    p0 = np.array([getattr(model, n) for n in names], dtype=float)

    def resid(p):
        m = model.with_params(names, p)
        return (m.pixel_to_enu(x, y) - v_target).ravel()

    r = least_squares(resid, p0, loss="soft_l1", f_scale=math.radians(0.15), x_scale="jac", max_nfev=300)
    return model.with_params(names, r.x)


def blind_search(model0: CameraModel, x, y, v_cat, tol_deg=1.5,
                 scales=(0.85, 0.92, 1.0, 1.08, 1.16), roll_step=2.0, mirrors=(False, True)):
    """粗搜索 roll / 镜像 / 焦距尺度。返回 (最佳模型, 匹配数)。"""
    best = (model0, -1)
    cos_tol = math.cos(math.radians(tol_deg))
    rolls = np.radians(np.arange(0, 360, roll_step))
    for mirror in mirrors:
        for s in scales:
            m = CameraModel.from_dict({**model0.to_dict(), "k1": model0.k1 * s, "k2": 0.0, "k3": 0.0,
                                       "mirror": mirror, "roll": 0.0, "tilt_x": 0.0, "tilt_y": 0.0})
            vc = m.pixel_to_enu(x, y)  # roll=0 时的方向
            for r in rolls:
                score = int(np.count_nonzero(np.max((vc @ _rz(r).T) @ v_cat.T, axis=1) >= cos_tol))
                if score > best[1]:
                    best = (CameraModel.from_dict({**m.to_dict(), "roll": math.degrees(r)}), score)
    return best


def calibrate(img, t_unix, lat, lon, init: CameraModel, mask=None, blind=True, max_mag=5.0,
              gain=1.0, min_stars=20, max_rms_arcmin=8.0, catalog: Catalog | None = None):
    """从一张夜空快照标定相机。返回 (CameraModel, report dict)。"""
    stars = detect_stars(img, mask)
    n_det = len(stars["x"])
    report = {"detected": n_det, "ok": False}
    if n_det < min_stars:
        report["error"] = f"检测到的星点太少 ({n_det} < {min_stars})，可能多云或曝光不足"
        return init, report
    cat = (catalog or load_catalog()).brighter(max_mag)
    v_all, idx_all = catalog_enu(cat, t_unix, lat, lon, min_alt=3.0)
    mag_all = cat.mag[idx_all]
    x, y = stars["x"], stars["y"]

    model = init
    if blind:
        nb = min(80, n_det)
        bright = mag_all <= min(4.0, max_mag)
        model, score = blind_search(init, x[:nb], y[:nb], v_all[bright])
        report["blind_score"] = score
        log.info("粗搜索: roll=%.1f mirror=%s k1=%.4f 匹配=%d", model.roll, model.mirror, model.k1, score)

    stages = [(2.0, ("x0", "y0", "k1", "roll"), 4.0), (1.2, ("x0", "y0", "k1", "k2", "roll"), 4.5),
              (0.8, PARAM_NAMES, max_mag), (0.5, PARAM_NAMES, max_mag), (0.35, PARAM_NAMES, max_mag),
              (0.35, PARAM_NAMES, max_mag)]
    di = ci = np.zeros(0, int)
    for tol, names, mlim in stages:
        sel = mag_all <= mlim
        v_cat = v_all[sel]
        di, ci_sel = match_nearest(model.pixel_to_enu(x, y), v_cat, tol)
        ci = np.nonzero(sel)[0][ci_sel]
        if len(di) < max(8, len(names) + 2):
            report["error"] = f"匹配星不足 ({len(di)})"
            return init, report
        if len(di) < 40:  # 星少时不拟合高阶项，避免过拟合
            names = tuple(n for n in names if n not in ("k3", "tilt_x", "tilt_y"))
        model = _fit(model, x[di], y[di], v_all[ci], names)

    resid = astro.vec_angle(model.pixel_to_enu(x[di], y[di]), v_all[ci]) * 60.0
    rms = float(np.sqrt(np.mean(resid**2)))
    flux = stars["flux"][di]
    unsat = (stars["peak"][di] < 250) & (flux > 0)
    zp = None
    if np.count_nonzero(unsat) >= 5:
        zp = float(np.median(mag_all[ci][unsat] + 2.5 * np.log10(flux[unsat])))
    model.zp, model.zp_gain = zp, float(gain)
    model.rms_arcmin, model.n_stars = rms, int(len(di))
    model.fitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_unix))
    report.update(matched=int(len(di)), rms_arcmin=rms, zp=zp,
                  ok=bool(len(di) >= min_stars and rms <= max_rms_arcmin),
                  matches={"det": di.tolist(), "cat": idx_all[ci].tolist()})
    if not report["ok"]:
        report["error"] = f"标定质量不足: 匹配 {len(di)} 颗, rms {rms:.1f}′"
    return model, report


# ---------------------------------------------------------------- 可视化
def draw_overlay(img, model: CameraModel, t_unix=None, lat=None, lon=None, stars=None,
                 catalog: Catalog | None = None, max_mag=3.5):
    """在图像上绘制高度/方位网格、方位字母，以及可选的星表位置，用于人工核对标定。"""
    if img.ndim == 2:
        img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = img.copy()
    grid = (90, 160, 255)
    for alt in (0, 30, 60):
        az = np.linspace(0, 360, 361)
        px, py = model.altaz_to_pixel(np.full_like(az, alt), az)
        cv2.polylines(out, [np.stack([px, py], 1).astype(np.int32)], True, grid, 1, cv2.LINE_AA)
    for az in range(0, 360, 45):
        alt = np.linspace(0, 90, 91)
        px, py = model.altaz_to_pixel(alt, np.full_like(alt, az))
        cv2.polylines(out, [np.stack([px, py], 1).astype(np.int32)], False, grid, 1, cv2.LINE_AA)
    for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        px, py = model.altaz_to_pixel(8.0, az)
        cv2.putText(out, label, (int(px) - 8, int(py) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
                    cv2.LINE_AA)
    if stars is not None:
        for sx, sy in zip(stars["x"], stars["y"]):
            cv2.circle(out, (int(round(sx)), int(round(sy))), 6, (0, 255, 0), 1, cv2.LINE_AA)
    if catalog is not None and t_unix is not None:
        cat = catalog.brighter(max_mag)
        v, idx = catalog_enu(cat, t_unix, lat, lon, min_alt=2.0)
        px, py = model.enu_to_pixel(v)
        for x, y, i in zip(px, py, idx):
            cv2.drawMarker(out, (int(x), int(y)), (60, 60, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            if cat.mag[i] < 1.5:
                cv2.putText(out, cat.name[i], (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (60, 60, 255), 1, cv2.LINE_AA)
    return out
