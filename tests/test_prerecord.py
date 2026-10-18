import cv2
import numpy as np

from starchaser.prerecord import EncodedRing, FrameRing


def test_frame_ring_writes_pre_roll(tmp_path):
    ring = FrameRing(keep_seconds=4.0, fps=10)
    for i in range(50):  # 5 秒
        ring.push(np.full((48, 64, 3), i, np.uint8), i / 10)
    ring.begin(tmp_path / "clip", pre_seconds=2.0, now_ts=4.9)
    for i in range(50, 60):
        ring.push(np.full((48, 64, 3), i, np.uint8), i / 10)
    info = ring.end()
    assert info["frames"] == 21 + 10            # 2 秒预录 (含当前帧) + 1 秒续录
    assert info["frame_ts"][0] == 2.9
    cap = cv2.VideoCapture(str(tmp_path / info["file"]))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == info["frames"]


def test_encoded_ring_starts_on_keyframe(tmp_path):
    ring = EncodedRing(keep_seconds=3.0, fps=10)
    for i in range(60):  # 每 10 帧一个关键帧，时间戳为微秒
        ring.outputframe(bytes([i]), keyframe=(i % 10 == 0), timestamp=i * 100_000)
    ring.begin(tmp_path / "clip", pre_seconds=1.5, now_ts=1000.0)
    for i in range(60, 65):
        ring.outputframe(bytes([i]), keyframe=False, timestamp=i * 100_000)
    info = ring.end()
    data = (tmp_path / info["file"]).read_bytes()
    # 最新帧 59 → 截止 44 → 不晚于它的最后一个关键帧是 40
    assert data[0] == 40 and data[-1] == 64
    assert info["frames"] == 25
    assert abs(info["frame_ts"][19] - 1000.0) < 1e-6   # 第 59 帧对应 begin 时的墙钟
