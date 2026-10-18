import calendar

import numpy as np
import pytest

from starchaser import astro


def utc(*a):
    return calendar.timegm(tuple(a) + (0,) * (6 - len(a)))


def test_gmst_meeus_12b():
    # Meeus 例 12.b: 1987-04-10 19:21:00 UT → 8h34m57.0896s
    assert astro.gmst_deg(2446896.30625) == pytest.approx(128.7378734, abs=1e-6)


def test_precession_meeus_21b():
    # Meeus 例 21.b: θ Per (已含自行) 岁差到 2028-11-13.19
    ra, dec = astro.j2000_to_date(41.054063, 49.227750, 2462088.69)
    assert ra == pytest.approx(41.547214, abs=2e-5)
    assert dec == pytest.approx(49.348483, abs=2e-5)
    back = astro.date_to_j2000(ra, dec, 2462088.69)
    assert back[0] == pytest.approx(41.054063, abs=1e-8) and back[1] == pytest.approx(49.227750, abs=1e-8)


def test_sun_equinox_and_solar_longitude():
    lam, _ = astro.sun_ecliptic_lon(astro.jd_from_unix(utc(2024, 3, 20, 3, 6)))
    assert abs((lam + 180) % 360 - 180) < 0.02
    # 英仙座极大 λ☉≈140.0° 在 8 月 12-13 日
    assert 139.0 < astro.solar_longitude_j2000(utc(2026, 8, 12, 18)) < 141.0


def test_sun_altitude_noon_and_midnight():
    lat, lon = 31.23, 121.47
    # 上海地方正午 (约 UTC 03:55) 夏至太阳高度 ≈ 90 - 31.23 + 23.44
    assert astro.sun_altitude(utc(2026, 6, 21, 3, 55), lat, lon) == pytest.approx(82.2, abs=0.3)
    assert astro.sun_altitude(utc(2026, 6, 21, 15, 55), lat, lon) < -30


def test_altaz_roundtrip_and_refraction():
    ra = np.array([10.0, 100.0, 250.0])
    dec = np.array([-20.0, 20.0, 70.0])
    alt, az = astro.radec_to_altaz(ra, dec, 31.0, 123.0)
    ra2, dec2 = astro.altaz_to_radec(alt, az, 31.0, 123.0)
    assert np.allclose(ra2, ra) and np.allclose(dec2, dec)
    # 地平折射约 34′，10° 处约 5.3′；两个方向的公式互逆 (误差 < 0.1′)
    assert -astro.refraction_apparent_to_true(0.0) * 60 == pytest.approx(34.5, abs=0.5)
    for h in (5.0, 10.0, 45.0):
        assert astro.refraction_apparent_to_true(astro.refraction_true_to_apparent(h)) == pytest.approx(h, abs=0.002)


def test_apparent_altaz_j2000_roundtrip():
    t = utc(2026, 10, 21, 12)
    alt, az = astro.j2000_to_apparent_altaz(279.23458, 38.78361, t, 31.23, 121.47)  # 织女星，傍晚高约 50°
    assert 30 < alt < 70
    ra, dec = astro.apparent_altaz_to_j2000(alt, az, t, 31.23, 121.47)
    assert astro.angular_sep(ra, dec, 279.23458, 38.78361) < 0.005


def test_sun_crossings_order():
    ev = astro.sun_crossings(utc(2026, 10, 18, 4), 24, 31.23, 121.47, -10.0)
    kinds = [k for _, k in ev]
    assert kinds == ["set", "rise"]
    dark_h = (ev[1][0] - ev[0][0]) / 3600
    assert 9 < dark_h < 12
