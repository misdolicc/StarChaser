import numpy as np

from starchaser.config import load_config
from starchaser.detector import RealtimeDetector
from starchaser.synth import add_gaussian

H, W, FPS = 180, 320, 25.0


def run(scene, n=150, seed=0):
    """scene(fi, img) 在帧上绘制目标；返回 (确认次数, 是否出现全局事件)。"""
    cfg = load_config().detector
    det = RealtimeDetector(cfg, (H, W), fps=FPS)
    rng = np.random.default_rng(seed)
    stars = rng.integers(0, [W, H], size=(40, 2))
    confirmed, global_ev = [], False
    for fi in range(n):
        img = np.full((H, W), 30.0, np.float32)
        for k, (x, y) in enumerate(stars):  # 闪烁的恒星
            add_gaussian(img, x, y, 400 * (1 + 0.3 * np.sin(fi * 0.7 + k)), 1.0)
        scene(fi, img)
        img += rng.normal(0, 2.5, img.shape)
        r = det.process(np.clip(img, 0, 255).astype(np.uint8), fi, fi / FPS)
        confirmed += r.new_confirmed
        global_ev |= r.global_event
    return confirmed, global_ev


def test_meteor_confirmed():
    def scene(fi, img):
        if 80 <= fi < 95:
            add_gaussian(img, 60 + 4.0 * (fi - 80), 40 + 2.0 * (fi - 80), 900, 1.2)
    conf, _ = run(scene)
    assert len(conf) == 1
    t = conf[0]
    assert t.first.fi <= 83 and abs(t.first.x - 60) < 3


def test_static_and_slow_objects_ignored():
    def scene(fi, img):
        add_gaussian(img, 200 + 0.03 * fi, 100, 700, 1.2)          # 卫星级慢速目标
        if (fi // 12) % 2:
            add_gaussian(img, 100, 140, 900, 1.2)                     # 原地闪烁 (飞机频闪/恒星)
    conf, _ = run(scene, n=250)
    assert conf == []


def test_global_flash_rejected():
    def scene(fi, img):
        if fi in (100, 101):
            img += 40  # 闪电 / 车灯照亮全天
    conf, global_ev = run(scene)
    assert global_ev and conf == []
