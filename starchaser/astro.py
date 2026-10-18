"""天文计算：时间/恒星时、坐标转换、岁差、大气折射、太阳/月亮。

只依赖 numpy，精度目标 ~0.01°，远高于全天相机 ~0.1°/像素 的分辨率。
约定：角度一律用度；方位角从正北起向东量；时间用 Unix 秒 (UTC)。
"""
from __future__ import annotations

import math

import numpy as np

D2R = math.pi / 180.0
R2D = 180.0 / math.pi
UNIX_EPOCH_JD = 2440587.5
J2000 = 2451545.0


# ---------------------------------------------------------------- 时间
def jd_from_unix(t):
    return np.asarray(t, dtype=float) / 86400.0 + UNIX_EPOCH_JD


def unix_from_jd(jd):
    return (np.asarray(jd, dtype=float) - UNIX_EPOCH_JD) * 86400.0


def gmst_deg(jd):
    """格林尼治平恒星时 (Meeus 12.4)。UT1 ≈ UTC。"""
    jd = np.asarray(jd, dtype=float)
    d = jd - J2000
    T = d / 36525.0
    g = 280.46061837 + 360.98564736629 * d + 0.000387933 * T * T - T**3 / 38710000.0
    return np.mod(g, 360.0)


def lst_deg(jd, lon_deg):
    return np.mod(gmst_deg(jd) + lon_deg, 360.0)


# ---------------------------------------------------------------- 向量
def radec_to_vec(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


def vec_to_radec(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    ra = np.mod(np.degrees(np.arctan2(v[..., 1], v[..., 0])), 360.0)
    dec = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    return ra, dec


def angular_sep(ra1, dec1, ra2, dec2):
    v1 = radec_to_vec(ra1, dec1)
    v2 = radec_to_vec(ra2, dec2)
    return vec_angle(v1, v2)


def vec_angle(v1, v2):
    """两单位向量夹角 (度)，小角度下数值稳定。"""
    cross = np.linalg.norm(np.cross(v1, v2), axis=-1)
    dot = np.sum(np.asarray(v1) * np.asarray(v2), axis=-1)
    return np.degrees(np.arctan2(cross, dot))


# ---------------------------------------------------------------- 地平坐标
def radec_to_altaz(ra, dec, lat, lst):
    H = np.radians(np.asarray(lst) - np.asarray(ra))
    d = np.radians(dec)
    p = math.radians(lat)
    sin_alt = math.sin(p) * np.sin(d) + math.cos(p) * np.cos(d) * np.cos(H)
    x = math.cos(p) * np.sin(d) - math.sin(p) * np.cos(d) * np.cos(H)  # cos(alt)·cos(A)
    y = -np.cos(d) * np.sin(H)                                        # cos(alt)·sin(A)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    az = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    return alt, az


def altaz_to_radec(alt, az, lat, lst):
    a = np.radians(alt)
    A = np.radians(az)
    p = math.radians(lat)
    sin_d = math.sin(p) * np.sin(a) + math.cos(p) * np.cos(a) * np.cos(A)
    x = math.cos(p) * np.sin(a) - math.sin(p) * np.cos(a) * np.cos(A)  # cos(δ)·cos(H)
    y = -np.cos(a) * np.sin(A)                                        # cos(δ)·sin(H)
    H = np.degrees(np.arctan2(y, x))
    ra = np.mod(np.asarray(lst) - H, 360.0)
    dec = np.degrees(np.arcsin(np.clip(sin_d, -1.0, 1.0)))
    return ra, dec


def altaz_to_enu(alt, az):
    """地平坐标 → 本地 ENU 单位向量 (x=东, y=北, z=天顶)。"""
    a = np.radians(alt)
    A = np.radians(az)
    c = np.cos(a)
    return np.stack([c * np.sin(A), c * np.cos(A), np.sin(a)], axis=-1)


def enu_to_altaz(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    az = np.mod(np.degrees(np.arctan2(v[..., 0], v[..., 1])), 360.0)
    return alt, az


# ---------------------------------------------------------------- 岁差
def _r2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _r3(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def precession_matrix(jd: float) -> np.ndarray:
    """J2000 平赤道 → 瞬时平赤道 (IAU 1976, Meeus 21.2)。v_date = P @ v_j2000。"""
    t = (float(jd) - J2000) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t**3) / 3600.0 * D2R
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t**3) / 3600.0 * D2R
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t**3) / 3600.0 * D2R
    return _r3(-z) @ _r2(theta) @ _r3(-zeta)


def j2000_to_date(ra, dec, jd):
    v = radec_to_vec(ra, dec) @ precession_matrix(jd).T
    return vec_to_radec(v)


def date_to_j2000(ra, dec, jd):
    v = radec_to_vec(ra, dec) @ precession_matrix(jd)
    return vec_to_radec(v)


# ---------------------------------------------------------------- 大气折射
def refraction_true_to_apparent(alt):
    """真高度 → 视高度 (Saemundsson 公式，标准大气)。"""
    h = np.maximum(np.asarray(alt, dtype=float), -1.0)
    r = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))  # 角分
    return np.asarray(alt, dtype=float) + np.clip(r, 0.0, None) / 60.0


def refraction_apparent_to_true(alt):
    """视高度 → 真高度 (Bennett 公式)。"""
    h = np.maximum(np.asarray(alt, dtype=float), -1.0)
    r = 1.0 / np.tan(np.radians(h + 7.31 / (h + 4.4)))
    return np.asarray(alt, dtype=float) - np.clip(r, 0.0, None) / 60.0


# ---------------------------------------------------------------- 组合转换
def j2000_to_apparent_altaz(ra, dec, t_unix, lat, lon):
    """J2000 赤道坐标 → 观测者看到的 (含折射) 地平坐标。t_unix 为标量。"""
    jd = float(jd_from_unix(t_unix))
    ra_d, dec_d = j2000_to_date(ra, dec, jd)
    alt, az = radec_to_altaz(ra_d, dec_d, lat, lst_deg(jd, lon))
    return refraction_true_to_apparent(alt), az


def apparent_altaz_to_j2000(alt, az, t_unix, lat, lon):
    """视地平坐标 → J2000 赤道坐标。t_unix 可为与 alt 同形状的数组。"""
    t = np.asarray(t_unix, dtype=float)
    jd = jd_from_unix(t)
    alt_true = refraction_apparent_to_true(alt)
    ra_d, dec_d = altaz_to_radec(alt_true, az, lat, lst_deg(jd, lon))
    # 一个事件只持续数秒，岁差矩阵取平均时刻即可
    return date_to_j2000(ra_d, dec_d, float(np.mean(jd)))


# ---------------------------------------------------------------- 太阳 / 月亮
def sun_ecliptic_lon(jd):
    """太阳视黄经 (瞬时春分点) 与黄赤交角，低精度公式 (天文年历)，精度约 0.01°。"""
    n = np.asarray(jd, dtype=float) - J2000
    L = 280.460 + 0.9856474 * n
    g = np.radians(357.528 + 0.9856003 * n)
    lam = np.mod(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g), 360.0)
    eps = 23.439 - 0.0000004 * n
    return lam, eps


def sun_radec(jd):
    lam, eps = sun_ecliptic_lon(jd)
    lr, er = np.radians(lam), np.radians(eps)
    ra = np.mod(np.degrees(np.arctan2(np.cos(er) * np.sin(lr), np.cos(lr))), 360.0)
    dec = np.degrees(np.arcsin(np.sin(er) * np.sin(lr)))
    return ra, dec


def sun_altitude(t_unix, lat, lon):
    jd = jd_from_unix(t_unix)
    ra, dec = sun_radec(jd)
    alt, _ = radec_to_altaz(ra, dec, lat, lst_deg(jd, lon))
    return alt


def solar_longitude_j2000(t_unix):
    """J2000 春分点的太阳黄经 λ☉ —— 国际流星组织 (IMO) 流星雨日历使用的时间坐标。"""
    jd = jd_from_unix(t_unix)
    lam, _ = sun_ecliptic_lon(jd)
    years = (jd - J2000) / 365.25
    return np.mod(lam - 0.013969 * years, 360.0)


def moon_phase(t_unix):
    """返回 (月龄 天, 照亮比例 0-1)。平均朔望月近似，误差约半天。"""
    synodic = 29.530588853
    age = np.mod(jd_from_unix(t_unix) - 2451550.26, synodic)
    illum = (1 - np.cos(2 * math.pi * age / synodic)) / 2
    return float(age), float(illum)


def sun_crossings(t_start: float, hours: float, lat: float, lon: float, alt_deg: float):
    """在 [t_start, t_start+hours] 内搜索太阳高度穿越 alt_deg 的时刻。返回 [(t, 'set'|'rise')]。"""
    step = 600.0
    ts = t_start + np.arange(0, hours * 3600 + step, step)
    alts = sun_altitude(ts, lat, lon) - alt_deg
    out = []
    for i in np.nonzero(np.sign(alts[:-1]) != np.sign(alts[1:]))[0]:
        a, b = ts[i], ts[i + 1]
        fa = alts[i]
        for _ in range(20):  # 二分到 < 1 秒
            m = (a + b) / 2
            fm = float(sun_altitude(m, lat, lon)) - alt_deg
            if np.sign(fm) == np.sign(fa):
                a, fa = m, fm
            else:
                b = m
        out.append(((a + b) / 2, "set" if alts[i] > 0 else "rise"))
    return out
