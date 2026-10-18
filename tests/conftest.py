import calendar
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starchaser.calibration import CameraModel  # noqa: E402

LAT, LON = 31.23, 121.47


@pytest.fixture
def true_model():
    """带倾斜、畸变、任意朝向的"真实"镜头，测试中对系统未知。"""
    return CameraModel(960, 540, 483.7, 268.1, 270.0, math.radians(92.0), -0.03, 0.02,
                       tilt_x=1.5, tilt_y=-0.8, roll=217.0)


@pytest.fixture
def t_obs():
    return calendar.timegm((2026, 12, 13, 18, 0, 0))  # 北京时间 12-14 02:00，双子座流星雨
