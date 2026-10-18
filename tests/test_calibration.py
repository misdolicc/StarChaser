import numpy as np

from conftest import LAT, LON
from starchaser import astro, synth
from starchaser.calibration import CameraModel, calibrate


def test_pixel_enu_roundtrip(true_model):
    ys, xs = np.mgrid[60:480:40, 250:720:40].astype(float)
    x2, y2 = true_model.enu_to_pixel(true_model.pixel_to_enu(xs, ys))
    assert np.allclose(x2, xs, atol=1e-6) and np.allclose(y2, ys, atol=1e-6)


def test_initial_north_angle():
    m = CameraModel.initial(800, 800, 180, 400, north_angle=0)
    alt, az = m.pixel_to_altaz(400, 100)   # 图像正上方 → 北；r=300/400 → 天顶距 67.5°
    assert abs((az + 180) % 360 - 180) < 1e-6 and abs(alt - 22.5) < 1e-6
    _, az_e = m.pixel_to_altaz(100, 400)   # 左侧 → 东 (仰望天空)
    assert abs(az_e - 90) < 1e-6
    mm = CameraModel.initial(800, 800, 180, 400, north_angle=90, mirror=True)
    assert abs((mm.pixel_to_altaz(700, 400)[1] + 180) % 360 - 180) < 1e-6  # 北在右侧


def test_blind_calibration_recovers_lens(true_model, t_obs):
    img = synth.make_snapshot(true_model, t_obs, LAT, LON)
    init = CameraModel.initial(960, 540, fov_deg=185, circle_radius=270)  # 不知道方向、倾斜、畸变
    model, rep = calibrate(img, t_obs, LAT, LON, init, mask=true_model.sky_mask(0), gain=8.0)
    assert rep["ok"], rep
    assert rep["matched"] > 150
    assert abs(model.roll - true_model.roll) < 0.2 and not model.mirror
    ys, xs = np.mgrid[40:500:30, 230:740:30].astype(float)
    inside = np.hypot(xs - 484, ys - 268) < 250
    err = astro.vec_angle(true_model.pixel_to_enu(xs[inside], ys[inside]),
                          model.pixel_to_enu(xs[inside], ys[inside])) * 60
    assert np.median(err) < 1.5 and err.max() < 5.0   # 角分；像元约 20′
    assert 8.5 < model.zp < 9.8  # 渲染零点 9.4 (含大气消光)


def test_scaled_model_consistent(true_model):
    lo = true_model.scaled(480, 270)
    a1 = np.array(true_model.pixel_to_altaz(600.0, 200.0))
    a2 = np.array(lo.pixel_to_altaz(300.0, 100.0))
    assert np.allclose(a1, a2, atol=1e-9)
