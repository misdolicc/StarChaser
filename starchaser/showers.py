"""流星雨归属 (单站法)。

把流星轨迹反向延长成大圆，若某个活跃流星雨的辐射点:
    1. 距该大圆足够近 (容差随轨迹长度变短而放宽 —— 短轨迹方向误差大)
    2. 位于轨迹"后方" (流星远离辐射点运动)
    3. 与流星起点角距 < 90°，且辐射点在地平线附近以上
则归属该流星雨，否则记为偶发流星 (SPO)。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import astro

DATA = Path(__file__).resolve().parent / "data" / "showers.json"
_CACHE = None


def load_showers(path=None):
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    data = json.loads(Path(path or DATA).read_text(encoding="utf-8"))["showers"]
    if path is None:
        _CACHE = data
    return data


def _in_window(md: str, begin: str, end: str) -> bool:
    return begin <= md <= end if begin <= end else (md >= begin or md <= end)


def active_showers(t_unix, showers=None):
    """t 时刻活跃的流星雨及其当时的辐射点 (按太阳黄经线性漂移)。"""
    md = datetime.fromtimestamp(t_unix, timezone.utc).strftime("%m-%d")
    sol = float(astro.solar_longitude_j2000(t_unix))
    out = []
    for s in showers or load_showers():
        if not _in_window(md, *s["active"]):
            continue
        d = (sol - s["peak_sol"] + 180.0) % 360.0 - 180.0
        out.append({**s, "ra_now": (s["ra"] + s["dra"] * d) % 360.0,
                    "dec_now": float(np.clip(s["dec"] + s["ddec"] * d, -90, 90)), "dsol": d})
    return out


def associate(ra0, dec0, ra1, dec1, t_unix, lat, lon, showers=None, pos_sigma_deg=0.15, base_tol_deg=3.0):
    """返回 (代码, 辐射点到大圆的角距°)，偶发流星返回 ("SPO", None)。"""
    v0 = astro.radec_to_vec(ra0, dec0)
    v1 = astro.radec_to_vec(ra1, dec1)
    length = float(astro.vec_angle(v0, v1))
    if length < 0.3:
        return "SPO", None
    pole = np.cross(v0, v1)
    pole /= np.linalg.norm(pole)
    vmid = (v0 + v1) / np.linalg.norm(v0 + v1)
    best = ("SPO", None, math.inf)
    jd = float(astro.jd_from_unix(t_unix))
    lst = float(astro.lst_deg(jd, lon))
    for s in active_showers(t_unix, showers):
        r = astro.radec_to_vec(s["ra_now"], s["dec_now"])
        ra_d, dec_d = astro.j2000_to_date(s["ra_now"], s["dec_now"], jd)
        r_alt, _ = astro.radec_to_altaz(ra_d, dec_d, lat, lst)
        if r_alt < -5.0:
            continue
        d_gc = math.degrees(abs(math.asin(float(np.clip(np.dot(pole, r), -1, 1)))))
        d0 = float(astro.vec_angle(r, v0))
        d1 = float(astro.vec_angle(r, v1))
        if not (d0 < d1) or d0 > 90.0:
            continue
        # 方向误差 ≈ σ/L (弧度)，外推到辐射点 D 处的偏差 ≈ D·σ/L
        dist_mid = float(astro.vec_angle(r, vmid))
        tol = min(10.0, base_tol_deg + 2.0 * dist_mid * pos_sigma_deg / max(length, 0.5))
        if d_gc <= tol and d_gc / tol < best[2]:
            best = (s["code"], round(d_gc, 2), d_gc / tol)
    return best[0], best[1]


def shower_names(showers=None):
    return {s["code"]: s["name"] for s in (showers or load_showers())} | {"SPO": "偶发"}
