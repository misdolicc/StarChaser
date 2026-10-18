import math

import numpy as np

from conftest import LAT, LON
from starchaser import astro
from starchaser.showers import active_showers, associate
from starchaser.synth import SynthMeteor


def meteor(radiant, dist=40.0, toward=False, offset_deg=0.0):
    """在距辐射点 dist° 处构造一颗流星，返回起止点 J2000 (ra0, dec0, ra1, dec1)。

    offset_deg: 让运动方向对应的"辐射点"偏离真实辐射点 (模拟不相关的流星)。
    """
    r = astro.radec_to_vec(*radiant)
    p = np.cross(r, [0.0, 0.0, 1.0])
    p /= np.linalg.norm(p)
    s = r * math.cos(math.radians(dist)) + p * math.sin(math.radians(dist))
    start = astro.vec_to_radec(s)
    fake_radiant = (radiant[0], radiant[1] + offset_deg)  # p 沿赤经方向，故在赤纬方向偏移
    m = SynthMeteor(0, 0.5, float(start[0]), float(start[1]), *fake_radiant, speed_deg_s=20.0)
    v0, v1 = m.vec_at(0), m.vec_at(0.5)
    if toward:
        v0, v1 = v1, v0
    return (*astro.vec_to_radec(v0), *astro.vec_to_radec(v1))


def gem(t):
    return next((s["ra_now"], s["dec_now"]) for s in active_showers(t) if s["code"] == "GEM")


def test_active_showers_dec14(t_obs):
    codes = {s["code"] for s in active_showers(t_obs)}
    assert "GEM" in codes
    assert "URS" not in codes and "PER" not in codes  # URS 12-17 才开始


def test_gem_meteor_associated(t_obs):
    code, dist = associate(*meteor(gem(t_obs)), t_obs, LAT, LON)
    assert code == "GEM" and dist < 0.5


def test_wrong_direction_or_offset_is_sporadic(t_obs):
    assert associate(*meteor(gem(t_obs), toward=True), t_obs, LAT, LON)[0] == "SPO"
    assert associate(*meteor(gem(t_obs), offset_deg=25), t_obs, LAT, LON)[0] == "SPO"
