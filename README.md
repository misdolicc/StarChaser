# StarChaser — 树莓派全天流星监测

部署在 **树莓派 Zero 2W + 广角/鱼眼全天相机** 上的流星自动监测系统：

- **24 小时无人值守**：按太阳高度自动进入/退出夜间监测，白天分析数据、整理报表、清理空间
- **预录像**：内存环形缓冲保存最近数秒的硬件 H.264 视频，触发时连同"过去" 3 秒一起写盘，流星出现的第一帧也不会丢
- **两级检测**：低分辨率流上实时检测 (宽松、低开销) → 全分辨率片段离线精确判别 (排除飞机、卫星、闪烁星、噪声)
- **关键帧**：最大值叠加图、标注图、起点/峰值/终点帧、逐帧小图、整夜流星合成图
- **流星位置解析**：全天镜头自动标定 (自动认星，无需手工)，输出每帧的高度/方位角与 J2000 赤经赤纬、角长度、角速度、估算星等，并判定所属流星雨
- **自动报表**：每晚生成 HTML 报表 (统计卡片、每小时流星数、全天轨迹图、星等分布、事件列表) 与 CSV/JSON，局域网浏览器直接查看

```
 全天相机 ──► ISP ─┬─ main 1080p ─► 硬件 H.264 ─► 环形缓冲 (≈5 s, 内存) ──触发──► inbox/<事件>/clip.h264
                   └─ lores 640×360 ─► 实时检测 (背景差分 + 航迹关联) ──┘          meta.json (lores 航迹)
                                                                                      │
   starchaser-worker (低优先级) ◄──────────────────────────────────────────────────────┘
     叠加 max/ave/std ─► (x,y,帧号) RANSAC 提取航迹 ─► 测光/关键帧 ─► 判别 (流星 / 排除)
     ─► 像素→高度方位→赤经赤纬 ─► 流星雨归属 ─► SQLite ─► report.html / CSV / 合成图
     每晚结束：用快照自动精修标定 ─► 重算天球坐标 ─► 清理空间
```

## 在 PC 上先跑一遍演示 (不需要相机)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy opencv-python-headless scipy pytest tzdata
python -m starchaser demo --data demo               # 合成双子座流星雨之夜 → 检测 → 盲标定 → 分析 → 报表
python -m pytest                                    # 22 项测试
```

打开 `demo/data/nights/2026-12-13/report.html` 查看报表。演示会生成一段带星空、流星、飞机、热像素和噪声的合成视频，再走完与树莓派上完全相同的流程。合成视频的真实参数已知，可以用来检验测量结果。

## 部署到树莓派

```bash
# Raspberry Pi OS Lite 64-bit (Bookworm)，相机已在 config.txt 中启用 (见 docs/HARDWARE.md)
git clone <本仓库> ~/StarChaser && cd ~/StarChaser
bash deploy/install.sh          # 安装 apt 依赖、/etc/starchaser.toml、三个 systemd 服务
nano /etc/starchaser.toml       # 至少填写站点经纬度、镜头视场
```

| 服务 | 作用 |
|---|---|
| `starchaser-capture` | 采集：实时检测 + 预录像 + 定时叠加快照；带 systemd 看门狗 |
| `starchaser-worker` | 处理：分析事件、天体测量、写数据库与报表；每晚结束后自动精修标定 |
| `starchaser-web` | 在 `http://<树莓派IP>:8080/` 提供报表 |

**首次标定**：第一个晴夜过后，处理服务会用星最多的快照自动做"盲标定" (只需配置中的经纬度，镜头朝向和畸变都会自动求出)。完成后可查看 `calibration/check_<日期>.jpg`，星表位置 (红色 ×) 应与星点 (绿色 ○) 重合。也可以手动执行：

```bash
python3 -m starchaser calibrate --night 2026-10-21    # 或 --image xxx.png --time 2026-10-21T14:00:00Z
python3 -m starchaser reprocess --night 2026-10-21    # 标定更新后重算该夜的天球坐标 (无需重新解码视频)
```

标定后会生成 `mask.png` (白色 = 检测区域)，可用任意画图软件把树木、建筑、路灯涂黑。

## 输出

```
starchaser-data/
├── index.html                      所有观测夜 + 实时系统状态
├── nights/2026-10-21/
│   ├── report.html                 当夜报表
│   ├── events.csv  summary.json
│   ├── composite.jpg               整夜流星合成图
│   ├── meteors/20261021T183512_120/
│   │   ├── clip.mp4                预录视频 (含触发前 3 秒)
│   │   ├── stack.jpg  annotated.jpg  thumb_1.jpg  montage_1.jpg
│   │   ├── frames/m1_start.jpg  m1_peak.jpg  m1_end.jpg
│   │   └── event.json              每帧像素位置、时间、高度/方位、赤经赤纬、测光
│   └── rejected/…                  被排除的候选 (只保留叠加图与原因)
├── snapshots/  calibration/  meteors.db (SQLite)
```

`event.json` 中 `measure` 字段含：起止时刻、持续时间、起止高度/方位角、起止赤经/赤纬 (J2000)、角长度、角速度、估算峰值星等、流星雨代码及辐射点到轨迹大圆的距离、每帧赤经赤纬。

## 常用命令

```bash
python3 -m starchaser capture --video test.avi --start 2026-10-21T14:00:00Z   # 用视频文件仿真采集
python3 -m starchaser worker --once      # 处理积压事件后退出
python3 -m starchaser report             # 重新生成所有报表
python3 -m starchaser synth out.avi --meteors 8 --start 2026-08-12T18:00:00Z   # 生成合成测试视频
journalctl -u starchaser-capture -f      # 实时日志
```

## 已验证与未验证

- 已在 PC (Windows, Python 3.11) 上验证：22 项单元/端到端测试；合成数据上的盲标定 (方向误差中位数 < 1′)、检测、角速度 (误差约 1–7%)、流星雨归属；报表渲染。
- **尚未在真实树莓派硬件上运行**：`PiCameraSource` (picamera2 调用、ISP 线性 gamma 调优)、硬件编码器挂接的 `EncodedRing`、systemd 服务。首次部署时请先手动运行 `python3 -m starchaser -v capture` 观察日志与状态页的帧率/丢帧。
- 实时检测在开发 PC 上约 2.2 ms/帧 (640×360，单线程)；Zero 2W 的 Cortex-A53 预计慢 5–10 倍，应能跑满 25fps，但需实测——若状态页帧率不足，把 `lores_size` 改为 `[480, 270]`。
- 星等为相对恒星测光的估算值 (视频压缩、饱和会带来误差)；流星雨参数表 (`starchaser/data/showers.json`) 为 IMO 日历的近似值，建议每年核对。单站观测无法得到三维轨道。

更多细节：[设计文档](docs/DESIGN.md) · [硬件与部署](docs/HARDWARE.md)
