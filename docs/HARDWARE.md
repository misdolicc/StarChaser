# 硬件选型与现场部署

## 物料清单

| 部件 | 推荐 | 说明 |
|---|---|---|
| 主机 | Raspberry Pi Zero 2W | 带散热片；Wi-Fi 仅 2.4 GHz |
| 相机 | **IMX462 (STARVIS) CSI 模块** (Arducam / InnoMaker 等) | 2.9 µm 大像元、高近红外响应，夜空灵敏度明显优于普通模块；备选 IMX290/IMX327、树莓派 HQ 相机 (IMX477) |
| 镜头 | M12 鱼眼，视场 ≥ 180°，焦距见下文 | 等距 (f-θ) 或等立体角投影均可，标定模型会自动拟合 |
| 排线 | Zero 专用 22-pin → 15-pin CSI 排线 | Zero 的 CSI 接口比标准树莓派窄 |
| 存储 | 64–128 GB 高耐久 (High Endurance) microSD | 每个事件约 6–10 MB (8 Mbps 视频) + 1–2 MB 图片 |
| 电源 | 5 V 2.5 A；线长 > 3 m 时用 12 V 供电 + 靠近树莓派的 DC-DC 降压 | 欠压会降频，状态页 / `vcgencmd get_throttled` 可查 |
| 外壳 | 防水盒 + 亚克力/玻璃透明半球罩 | 球罩中心对准镜头，减少反射与畸变 |
| 防露 | 5 V 加热片 1–2 W 贴在球罩根部 + 干燥剂 | 结露是全天相机最常见的故障 |
| 可选 | DS3231 RTC 或 GPS 模块 | 离线站点保证时间准确 (天体测量依赖时钟) |

## 镜头焦距怎么选

要让 180° 的像圈完整落在传感器的短边内，等距鱼眼的焦距约为

```
f ≈ (传感器短边 / 2) / (π / 2)
```

| 传感器 / 模式 | 有效短边 | 建议焦距 | 配置 |
|---|---|---|---|
| IMX462/IMX290，1920×1080，2.9 µm | 3.13 mm | ≈ 1.0 mm (1.0–1.2 mm 鱼眼) | `main_size = [1920, 1080]` |
| IMX477 HQ，2028×1520 合并模式，3.1 µm | 4.71 mm | ≈ 1.5 mm (常见 1.55 mm 鱼眼) | `main_size = [1440, 1080]`，`lores_size = [480, 360]` |

焦距略长会裁掉地平附近 (影响不大，低空本来就受光污染)；略短则像圈变小、分辨率下降。将实际像圈半径 (像素) 填入 `calibration.circle_radius_px` 可加快首次标定。

## 系统设置

1. 用 Raspberry Pi Imager 写入 **Raspberry Pi OS Lite (64-bit)**，预设 Wi-Fi、SSH、用户名、**时区**。
2. 编辑 `/boot/firmware/config.txt` 启用相机 (以模块厂商文档为准)：

   ```ini
   # IMX462 / IMX290 模块 (使用 imx290 驱动)
   camera_auto_detect=0
   dtoverlay=imx290,clock-frequency=74250000
   # HQ 相机 (IMX477) 一般可自动识别，保持 camera_auto_detect=1 即可
   ```

3. `rpicam-hello --list-cameras` 确认识别到相机。
4. 增大交换空间 (512 MB 内存余量不大)：`/etc/dphys-swapfile` 中设 `CONF_SWAPSIZE=512`，然后 `sudo systemctl restart dphys-swapfile`。
5. `bash deploy/install.sh`，编辑 `/etc/starchaser.toml` 填写经纬度、镜头视场。

## 现场安装

1. **对焦**：晚上拍一张长曝光，放大检查星点是否最小最圆：
   `rpicam-still -t 1 --shutter 2000000 --gain 8 --awbgains 1,1 -o focus.jpg`
   M12 镜头拧到最佳位置后用螺丝或胶固定。
2. **朝向**：镜头大致朝天顶即可，北方大致朝图像上方更直观，但都不是必需的——倾斜与方向由自动标定求出。
3. **遮挡**：首次标定后编辑 `mask.png`，把树木、屋檐、路灯等涂黑，可大幅减少误触发。
4. **验证**：第一晚查看首页状态 (帧率应接近 25、丢帧为 0、CPU 温度 < 75 °C)；次日查看 `calibration/check_*.jpg` 与报表。

## 故障排查

| 现象 | 可能原因 / 处理 |
|---|---|
| 状态页帧率明显低于配置值 | CPU 不足：`lores_size` 改为 `[480, 270]` 或 `fps = 20`；确认处理进程设了 `process_at_night = false` 后是否改善 |
| 启动报缓冲区分配失败 | 降低 `camera.buffer_count` (如 4)，或在 config.txt 中增大 CMA |
| 夜里误触发很多 | 雨滴/昆虫/灯光：完善 `mask.png`，调高 `detector.k_sigma`；被排除的候选不会进入流星列表 |
| 标定失败 "星点太少" | 多云、曝光不足或对焦不准；换一个晴夜，或调高 `max_gain` |
| 标定残差大 (> 8′) | 时间不准 (检查 NTP)、经纬度填错、镜头松动 |
| 赤经赤纬整体偏移 | 系统时间偏差：1 秒误差 ≈ 0.004°，1 分钟 ≈ 0.25° |
| 视频是 .h264 而不是 .mp4 | 未安装 ffmpeg：`sudo apt install ffmpeg` (不影响分析) |
