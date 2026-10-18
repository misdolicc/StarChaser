"""系统工具：systemd 通知/看门狗、CPU 温度、NTP 同步状态。"""
from __future__ import annotations

import os
import socket
import subprocess


def sd_notify(msg: str) -> bool:
    """向 systemd 发送通知 (READY=1 / WATCHDOG=1 / STATUS=...)。非 systemd 环境下静默忽略。"""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr or not hasattr(socket, "AF_UNIX"):
        return False
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
        return True
    except OSError:
        return False


def cpu_temp() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def throttled() -> str | None:
    """vcgencmd get_throttled：非 0 表示欠压/过热降频，需检查电源与散热。"""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout
        return out.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def ntp_synced() -> bool | None:
    """时间是否已同步 —— 天体测量依赖准确时钟 (树莓派无 RTC，断网时需 RTC/GPS 模块)。"""
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        return out == "yes" if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def set_nice(n: int):
    try:
        os.nice(n)
    except (AttributeError, OSError):
        pass
