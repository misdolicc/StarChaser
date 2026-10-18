#!/usr/bin/env bash
# StarChaser 安装脚本 —— 在树莓派 Zero 2W (Raspberry Pi OS Lite 64-bit, Bookworm) 上运行:
#   cd ~/StarChaser && bash deploy/install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"

echo "==> 安装系统依赖 (全部来自 apt，避免在 Zero 上编译)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-picamera2 python3-opencv python3-scipy python3-numpy ffmpeg

echo "==> 检查相机"
if ! rpicam-hello --list-cameras 2>/dev/null | grep -q ":"; then
    echo "!! 没有检测到相机。请检查排线方向与 /boot/firmware/config.txt 中的 dtoverlay (见 docs/HARDWARE.md)"
fi

echo "==> 配置文件 /etc/starchaser.toml"
if [ ! -f /etc/starchaser.toml ]; then
    sudo cp "$APP_DIR/config/starchaser.example.toml" /etc/starchaser.toml
    sudo chown "$USER_NAME" /etc/starchaser.toml
    echo "   已创建，请编辑站点经纬度与相机参数: nano /etc/starchaser.toml"
else
    echo "   已存在，保留不变"
fi

echo "==> 安装 systemd 服务"
for unit in capture worker web; do
    sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$APP_DIR|g" \
        "$APP_DIR/deploy/starchaser-$unit.service" | sudo tee "/etc/systemd/system/starchaser-$unit.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable starchaser-capture starchaser-worker starchaser-web
sudo systemctl restart starchaser-capture starchaser-worker starchaser-web

SWAP=$(free -m | awk '/Swap/ {print $2}')
if [ "${SWAP:-0}" -lt 400 ]; then
    echo "!! 交换空间仅 ${SWAP}MB。建议在 /etc/dphys-swapfile 设置 CONF_SWAPSIZE=512 后执行"
    echo "   sudo systemctl restart dphys-swapfile   (512MB 内存的 Zero 2W 需要缓冲)"
fi
if ! timedatectl show -p NTPSynchronized --value | grep -q yes; then
    echo "!! 系统时间尚未同步。天体测量依赖准确时间；离线部署请加装 RTC 或 GPS 模块"
fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "完成。查看日志: journalctl -u starchaser-capture -f"
echo "浏览报表:   http://${IP}:8080/"
