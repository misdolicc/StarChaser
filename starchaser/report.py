"""报表：每晚 HTML 报表 (统计卡片、每小时流星数、全天轨迹图、星等分布、事件表) + CSV/JSON + 整夜合成图，
以及总索引页。纯静态文件、无外部依赖，局域网内用浏览器直接打开 (或 `starchaser web`)。
"""
from __future__ import annotations

import csv
import html
import json
import logging
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from . import __version__, astro
from .showers import active_showers, shower_names
from .storage import Paths, write_json_atomic

log = logging.getLogger(__name__)

SERIES = ["var(--s1)", "var(--s2)", "var(--s3)"]
OTHER = "var(--muted)"


# ---------------------------------------------------------------- 时间
def get_tz(name: str, lon: float | None = None):
    """报表时区：配置的 IANA 时区 → (不可用时) 按经度取整的固定时区 → 系统本地时区。"""
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:
            log.warning("时区 %s 不可用 (Windows 需 pip install tzdata)", name)
            if lon is not None:
                return timezone(timedelta(hours=round(lon / 15.0)))
    return datetime.now().astimezone().tzinfo


def fmt_duration(seconds):
    if seconds is None:
        return "—"
    if seconds < 120:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟"
    return f"{seconds / 3600:.1f} 小时"


def fmt_time(t, tz, seconds=False):
    if t is None:
        return "—"
    return datetime.fromtimestamp(t, tz).strftime("%H:%M:%S" if seconds else "%H:%M")


def night_window(night: str, cfg):
    """当夜监测窗口 (太阳低于阈值的时段)，返回 (开始, 结束) Unix 时间或 (None, None)。"""
    lat, lon = cfg.site.latitude, cfg.site.longitude
    d = datetime.strptime(night, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    noon = (d + timedelta(hours=12 - lon / 15.0)).timestamp()
    ev = astro.sun_crossings(noon, 24, lat, lon, cfg.schedule.night_sun_altitude)
    start = next((t for t, k in ev if k == "set"), None)
    end = next((t for t, k in ev if k == "rise" and (start is None or t > start)), None)
    return start, end


# ---------------------------------------------------------------- 小工具
def esc(s):
    return html.escape(str(s), quote=True)


def num(v, fmt="{:.1f}", dash="—"):
    if v is None:
        return dash
    digits = int(fmt.split(".")[1][0]) if "." in fmt else 0
    return fmt.format(round(v, digits) + 0.0)  # 先舍入再 +0.0，避免显示 "-0.0"


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
a{color:var(--accent)}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.sub{color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.tile,.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .l{color:var(--ink2);font-size:13px}.tile .v{font-size:26px;font-weight:600;margin-top:2px}
.tile .d{color:var(--muted);font-size:12px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;align-items:start}
.card h3{font-size:14px;margin:0 0 8px;font-weight:600}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--muted);font-size:11px}
svg .ink{fill:var(--ink2)}
svg .grid{stroke:var(--grid);stroke-width:1;fill:none}
svg .axis{stroke:var(--axis);stroke-width:1;fill:none}
svg .bar{fill:var(--s1)}svg .bar:hover{opacity:.75}
svg .trk{stroke-width:2;fill:none;stroke-linecap:round}svg .trk:hover{stroke-width:4}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;color:var(--ink2);font-size:12px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--ring);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap;vertical-align:middle}
th{color:var(--ink2);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--surface)}
td.n{text-align:right}
td img{width:96px;height:96px;object-fit:cover;border-radius:6px;display:block;background:#000}
.files a{margin-right:8px}
.composite{width:100%;max-width:720px;border-radius:10px;display:block;background:#000}
details summary{cursor:pointer;color:var(--ink2)}
.empty{color:var(--muted);padding:24px;text-align:center}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
footer{margin-top:40px;color:var(--muted);font-size:12px}
"""


def page(title, body):
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{body}"
            f'<footer>StarChaser {__version__} · 生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>'
            f"</main></body></html>")


# ---------------------------------------------------------------- 图表 (内联 SVG)
def column_chart(labels, values, tips, w=520, h=220, y_label=""):
    """单系列柱状图。柱宽 ≤24px、顶端 4px 圆角、基线方正；每根柱带原生悬停提示。"""
    pad_l, pad_r, pad_t, pad_b = 34, 8, 12, 26
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(values) if values and max(values) > 0 else 1
    step = max(1, math.ceil(vmax / 4))
    top = step * math.ceil(vmax / step)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(y_label)}">']
    for k in range(0, top + 1, step):
        y = pad_t + ph - ph * k / top
        out.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{k}</text>')
    n = max(1, len(values))
    slot = pw / n
    bw = min(24.0, slot * 0.7)
    every = max(1, math.ceil(n / 12))
    for i, (lab, v, tip) in enumerate(zip(labels, values, tips)):
        x = pad_l + slot * i + (slot - bw) / 2
        bh = ph * v / top
        y = pad_t + ph - bh
        if v > 0:
            r = min(4.0, bh, bw / 2)
            out.append(f'<path class="bar" d="M{x:.1f},{pad_t + ph:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}'
                       f'H{x + bw - r:.1f}Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f}V{pad_t + ph:.1f}Z">'
                       f"<title>{esc(tip)}</title></path>")
        # 透明命中区域比柱子大，便于悬停
        out.append(f'<rect x="{pad_l + slot * i:.1f}" y="{pad_t}" width="{slot:.1f}" height="{ph}" fill="transparent">'
                   f"<title>{esc(tip)}</title></rect>")
        if i % every == 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{h - 8}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" x2="{w - pad_r}" y1="{pad_t + ph}" y2="{pad_t + ph}"/></svg>')
    return "".join(out)


def _polar_xy(alt, az, cx, cy, R):
    """天顶在中心、地平在边缘 (等距)，北上东左 —— 与仰望天空时的全天图像一致。"""
    r = R * (90.0 - alt) / 90.0
    a = math.radians(az)
    return cx - r * math.sin(a), cy - r * math.cos(a)


def _gc_points(alt0, az0, alt1, az1, n=12):
    v0, v1 = astro.altaz_to_enu(alt0, az0), astro.altaz_to_enu(alt1, az1)
    ang = math.acos(max(-1.0, min(1.0, float(np.dot(v0, v1)))))
    if ang < 1e-6:
        return [(alt0, az0), (alt1, az1)]
    pts = []
    for k in range(n + 1):  # 大圆插值
        u = k / n
        v = (math.sin((1 - u) * ang) * v0 + math.sin(u * ang) * v1) / math.sin(ang)
        a, z = astro.enu_to_altaz(v)
        pts.append((float(a), float(z)))
    return pts


def sky_chart(meteors, colors, names, tz, radiants=(), size=420):
    cx = cy = size / 2
    R = size / 2 - 22
    out = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="全天流星轨迹图"><defs>']
    for i, c in enumerate(SERIES + [OTHER]):  # 每种颜色一个箭头 marker
        out.append(f'<marker id="ah{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
                   f'orient="auto"><path d="M0,0L10,5L0,10z" style="fill:{c}"/></marker>')
    out.append("</defs>")
    for alt in (0, 30, 60):
        r = R * (90 - alt) / 90
        out.append(f'<circle class="{"axis" if alt == 0 else "grid"}" cx="{cx}" cy="{cy}" r="{r:.1f}"/>')
        if alt:
            out.append(f'<text x="{cx + 3}" y="{cy - r - 3:.1f}">{alt}°</text>')
    for lab, az in (("北", 0), ("东", 90), ("南", 180), ("西", 270)):
        x, y = _polar_xy(-14, az, cx, cy, R)
        out.append(f'<text class="ink" x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{lab}</text>')
    for rad in radiants:
        if rad["alt"] > 0:
            x, y = _polar_xy(rad["alt"], rad["az"], cx, cy, R)
            out.append(f'<path class="axis" d="M{x - 5:.1f},{y:.1f}H{x + 5:.1f}M{x:.1f},{y - 5:.1f}V{y + 5:.1f}" '
                       f'style="stroke:var(--ink2);stroke-width:1.5"><title>{esc(rad["name"])}座流星雨辐射点 (午夜)</title></path>'
                       f'<text class="ink" x="{x + 7:.1f}" y="{y - 5:.1f}">{esc(rad["code"])}</text>')
    for m in meteors:
        if m.get("alt0") is None:
            continue
        pts = [_polar_xy(a, z, cx, cy, R) for a, z in _gc_points(m["alt0"], m["az0"], m["alt1"], m["az1"])]
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        tip = (f"{fmt_time(m['t_start'], tz, True)} · {names.get(m.get('shower') or 'SPO', m.get('shower'))} · "
               f"{num(m.get('length_deg'))}° · {num(m.get('speed_deg_s'))}°/s"
               + (f" · {m['peak_mag']:+.1f} 等" if m.get("peak_mag") is not None else ""))
        c = colors(m)
        out.append(f'<path class="trk" d="{d}" style="stroke:{c}" marker-end="url(#ah{(SERIES + [OTHER]).index(c)})">'
                   f"<title>{esc(tip)}</title></path>")
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 整夜合成
def build_composite(paths: Paths, db, night):
    """整夜流星合成图：中间时刻的快照作底图 + 每颗流星的轨迹层 (trail.png) 取最大值叠加。"""
    nd = paths.night_dir(night)
    events = sorted((nd / "meteors").glob("*/event.json"))
    if not events:
        return None
    base = None
    snaps = sorted((paths.snapshots / night).glob("*.png"))
    if snaps:
        img = cv2.imread(str(snaps[len(snaps) // 2]), cv2.IMREAD_UNCHANGED)
        if img is not None:
            g = img.astype(np.float32) / (256.0 if img.dtype == np.uint16 else 1.0)
            h, w = g.shape
            center = g[h // 3:2 * h // 3, w // 3:2 * w // 3]
            lo = float(np.median(center))  # 天空背景压到接近黑色，只保留星点
            hi = max(float(np.percentile(center, 99.9)), lo + 1.0)
            g = np.clip((g - lo) / (hi - lo), 0, 1) ** 0.7 * 255
            base = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    roi = None
    for evf in events:
        ev = json.loads(evf.read_text(encoding="utf-8"))
        trail = cv2.imread(str(evf.parent / "trail.png"))
        if trail is None:
            continue
        w, h = ev["size"]
        x, y, rw, rh = ev["roi"]
        if base is None or base.shape[:2] != (h, w):
            base = np.zeros((h, w, 3), np.uint8) if base is None else cv2.resize(base, (w, h))
        region = base[y:y + rh, x:x + rw]
        region[:] = cv2.add(region, trail[:region.shape[0], :region.shape[1]])
        roi = (x, y, rw, rh)
    if base is None:
        return None
    if roi:
        x, y, rw, rh = roi
        base = base[y:y + rh, x:x + rw]
    out = nd / "composite.jpg"
    cv2.imwrite(str(out), base, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# ---------------------------------------------------------------- 每晚报表
def _sessions(nd: Path):
    f = nd / "sessions.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_night(paths: Paths, db, night: str, cfg, tz):
    nd = paths.night_dir(night)
    nd.mkdir(parents=True, exist_ok=True)
    meteors = db.meteors(night)
    rejects = db.rejects(night)
    names = shower_names()
    t_dusk, t_dawn = night_window(night, cfg)
    sessions = _sessions(nd)
    observed_s = sum(max(0.0, s["end"] - s["start"]) for s in sessions) if sessions else None
    mid = (t_dusk + t_dawn) / 2 if t_dusk and t_dawn else (meteors[0]["t_start"] if meteors else None)
    moon = astro.moon_phase(mid)[1] if mid else None
    calib = json.loads(paths.calib.read_text(encoding="utf-8")) if paths.calib.exists() else None

    # 颜色：数量最多的 3 个流星雨占用前三个分类色，其余 (含偶发) 归为灰色"其他"
    counts = Counter(m.get("shower") or "SPO" for m in meteors)
    top = [c for c, _ in counts.most_common() if c != "SPO"][:3]
    cmap = {c: SERIES[i] for i, c in enumerate(top)}

    def color(m):
        return cmap.get(m.get("shower") or "SPO", OTHER)

    # 统计卡片
    mags = [m["peak_mag"] for m in meteors if m.get("peak_mag") is not None]
    brightest = min(mags) if mags else None
    shower_txt = " · ".join(f"{names.get(c, c)} {n}" for c, n in counts.most_common(4)) or "—"
    tiles = [("流星", f"{len(meteors)}", shower_txt),
             ("排除的候选", f"{len(rejects)}", "飞机、卫星、噪声等"),
             ("有效监测", fmt_duration(observed_s),
              f"{fmt_time(t_dusk, tz)} – {fmt_time(t_dawn, tz)} 天黑窗口" if t_dusk else ""),
             ("最亮流星", f"{num(brightest, '{:+.1f}')} 等" if brightest is not None else "—", "估算星等 (仅供参考)"),
             ("月面照亮", f"{moon * 100:.0f}%" if moon is not None else "—", "月光会降低探测灵敏度")]
    body = [f"<h1>{esc(cfg.site.name)} · {night} 观测夜</h1>",
            f'<div class="sub">纬度 {cfg.site.latitude:.3f}° 经度 {cfg.site.longitude:.3f}° · '
            + (f"已标定 (匹配 {calib.get('n_stars', 0)} 颗星, 残差 {num(calib.get('rms_arcmin'))}′)" if calib else
               "<b>未标定</b>：只有像素坐标，没有天球坐标与流星雨归属")
            + ' · <a href="events.csv">CSV</a> · <a href="summary.json">JSON</a> · <a href="../../index.html">所有观测夜</a></div>',
            '<div class="tiles">' + "".join(
                f'<div class="tile"><div class="l">{esc(lab)}</div><div class="v">{esc(v)}</div>'
                f'<div class="d">{esc(d)}</div></div>' for lab, v, d in tiles) + "</div>"]

    # 每小时流星数
    charts = []
    if t_dusk and t_dawn:
        h0 = datetime.fromtimestamp(t_dusk, tz).replace(minute=0, second=0, microsecond=0)
        hours = []
        while h0.timestamp() < t_dawn:
            hours.append(h0)
            h0 += timedelta(hours=1)
    else:
        hours = sorted({datetime.fromtimestamp(m["t_start"], tz).replace(minute=0, second=0, microsecond=0)
                        for m in meteors})
    hc = Counter(datetime.fromtimestamp(m["t_start"], tz).replace(minute=0, second=0, microsecond=0) for m in meteors)
    if hours:
        charts.append('<div class="card"><h3>每小时流星数</h3>' + column_chart(
            [h.strftime("%H") for h in hours], [hc.get(h, 0) for h in hours],
            [f"{h.strftime('%H:00')}–{(h + timedelta(hours=1)).strftime('%H:00')} · {hc.get(h, 0)} 颗" for h in hours],
            y_label="每小时流星数") + "</div>")
    # 全天轨迹图
    if calib and meteors:
        rads = []
        if mid:
            for s in active_showers(mid):
                alt, az = astro.j2000_to_apparent_altaz(s["ra_now"], s["dec_now"], mid, cfg.site.latitude,
                                                        cfg.site.longitude)
                rads.append({"code": s["code"], "name": s["name"], "alt": float(alt), "az": float(az)})
        legend = "".join(f'<span><i style="background:{cmap[c]}"></i>{esc(names.get(c, c))} ({counts[c]})</span>'
                         for c in top)
        n_other = sum(n for c, n in counts.items() if c not in cmap)
        if n_other:
            legend += f'<span><i style="background:{OTHER}"></i>偶发及其他 ({n_other})</span>'
        charts.append('<div class="card"><h3>全天轨迹图 (北上东左，箭头为运动方向，＋为辐射点)</h3>'
                      + sky_chart(meteors, color, names, tz, rads) + f'<div class="legend">{legend}</div></div>')
    # 星等分布
    if mags:
        lo, hi = math.floor(min(mags)), math.ceil(max(mags) + 1e-9)
        bins = list(range(lo, max(hi, lo + 1)))
        mc = Counter(math.floor(m) for m in mags)
        charts.append('<div class="card"><h3>估算星等分布</h3>' + column_chart(
            [f"{b:+d}" for b in bins], [mc.get(b, 0) for b in bins],
            [f"{b:+d} ~ {b + 1:+d} 等 · {mc.get(b, 0)} 颗" for b in bins], y_label="星等分布") + "</div>")
    if charts:
        body.append('<h2>统计</h2><div class="charts">' + "".join(charts) + "</div>")
    if (nd / "composite.jpg").exists():
        body.append('<h2>整夜流星合成图</h2><a href="composite.jpg"><img class="composite" src="composite.jpg" '
                    'alt="整夜流星合成图"></a>')

    # 事件表 (同时是图表的表格视图)
    rows = []
    for i, m in enumerate(meteors, 1):
        d = Path(m["dir"]).relative_to(f"nights/{night}").as_posix()
        k = m["id"].rsplit("_", 1)[-1] if m["id"].count("_") > 1 else "1"
        ev_dir = paths.root / m["dir"]
        vid = next((p.name for p in ev_dir.glob("*") if p.suffix in (".mp4", ".h264", ".avi")), None)
        thumb = f"{d}/thumb_{k}.jpg"
        files = [f'<a href="{d}/annotated.jpg">标注图</a>', f'<a href="{d}/stack.jpg">叠加</a>',
                 f'<a href="{d}/frames/m{k}_peak.jpg">峰值帧</a>', f'<a href="{d}/montage_{k}.jpg">逐帧</a>']
        if vid:
            files.append(f'<a href="{d}/{vid}">视频</a>')
        files.append(f'<a href="{d}/event.json">数据</a>')
        sh = m.get("shower") or ("SPO" if m.get("calibrated") else None)
        rows.append(
            f"<tr><td class=n>{i}</td><td>{fmt_time(m['t_start'], tz, True)}</td>"
            f"<td class=n>{num(m.get('duration'), '{:.2f}')}</td>"
            f"<td class=n>{num(m.get('alt0'), '{:.0f}')}° / {num(m.get('az0'), '{:.0f}')}°</td>"
            f"<td class=n>{num(m.get('alt1'), '{:.0f}')}° / {num(m.get('az1'), '{:.0f}')}°</td>"
            f"<td class=n>{num(m.get('length_deg'))}</td><td class=n>{num(m.get('speed_deg_s'))}</td>"
            f"<td class=n>{num(m.get('peak_mag'), '{:+.1f}')}{' *' if m.get('saturated') else ''}</td>"
            f"<td><span class=dot style=\"background:{color(m)}\"></span>{esc(names.get(sh, sh) if sh else '—')}</td>"
            f'<td><a href="{d}/annotated.jpg"><img loading="lazy" src="{thumb}" alt="流星 {i}"></a></td>'
            f'<td class="files">{"".join(files)}</td></tr>')
    body.append("<h2>流星列表</h2>")
    if rows:
        body.append('<div class="tablewrap"><table><thead><tr><th>#</th><th>时间</th><th>时长 s</th>'
                    "<th>起点 高度/方位</th><th>终点 高度/方位</th><th>角长度 °</th><th>角速度 °/s</th>"
                    "<th>星等</th><th>流星雨</th><th>预览</th><th>文件</th></tr></thead><tbody>"
                    + "".join(rows) + "</tbody></table></div>"
                    '<p class="sub">星等由同夜恒星测光估算，* 表示饱和 (实际更亮)。方位角从正北向东量。</p>')
    else:
        body.append('<div class="empty">本夜没有确认的流星</div>')
    if rejects:
        rr = "".join(
            f"<tr><td>{fmt_time(r['t_start'], tz, True)}</td><td>{esc(r['reason'])}</td>"
            f'<td><a href="{Path(r["dir"]).relative_to(f"nights/{night}").as_posix()}/stack.jpg">叠加图</a></td></tr>'
            for r in rejects)
        body.append(f"<h2>排除的候选</h2><details><summary>{len(rejects)} 个候选被分析排除 (点击展开)</summary>"
                    f'<div class="tablewrap"><table><thead><tr><th>时间</th><th>原因</th><th>文件</th></tr></thead>'
                    f"<tbody>{rr}</tbody></table></div></details>")
    (nd / "report.html").write_text(page(f"{night} 流星观测报告", "".join(body)), encoding="utf-8")

    # CSV / JSON
    cols = ["id", "local_time", "t_start", "duration", "alt0", "az0", "alt1", "az1", "ra0", "dec0", "ra1", "dec1",
            "length_deg", "speed_deg_s", "peak_mag", "saturated", "shower", "shower_dist", "x0", "y0", "x1", "y1",
            "calibrated", "dir"]
    with (nd / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in meteors:
            m = {**m, "local_time": datetime.fromtimestamp(m["t_start"], tz).isoformat(timespec="milliseconds")}
            w.writerow([m.get(c) for c in cols])
    write_json_atomic(nd / "summary.json", {
        "night": night, "site": {"name": cfg.site.name, "lat": cfg.site.latitude, "lon": cfg.site.longitude},
        "meteors": len(meteors), "rejected": len(rejects), "showers": dict(counts), "brightest_mag": brightest,
        "observed_hours": round(observed_s / 3600, 3) if observed_s else None, "dark_window": [t_dusk, t_dawn], "moon_illumination": moon,
        "calibrated": bool(calib)})
    return nd / "report.html"


# ---------------------------------------------------------------- 总索引
def build_index(paths: Paths, db, cfg, tz):
    nights = db.nights()
    names = shower_names()
    rows = []
    for n in nights:
        ms = db.meteors(n["night"])
        top = Counter(m.get("shower") for m in ms if m.get("shower") and m["shower"] != "SPO").most_common(2)
        has = (paths.night_dir(n["night"]) / "report.html").exists()
        link = f'<a href="nights/{n["night"]}/report.html">{n["night"]}</a>' if has else n["night"]
        rows.append(f"<tr><td>{link}</td><td class=n>{n['meteors']}</td><td class=n>{n['rejects']}</td>"
                    f"<td>{esc(' · '.join(f'{names.get(c, c)} {k}' for c, k in top) or '—')}</td></tr>")
    recent = list(reversed(nights[:30]))
    body = [f"<h1>{esc(cfg.site.name)} 流星监测</h1>",
            f'<div class="sub">纬度 {cfg.site.latitude:.3f}° 经度 {cfg.site.longitude:.3f}° · 累计 '
            f'{sum(n["meteors"] for n in nights)} 颗流星 / {len(nights)} 个观测夜</div>',
            '<div class="tiles" id="status"><div class="tile"><div class="l">系统状态</div>'
            '<div class="v">—</div><div class="d">读取 status.json…</div></div></div>']
    if recent:
        body.append('<h2>近期每晚流星数</h2><div class="card">' + column_chart(
            [n["night"][5:] for n in recent], [n["meteors"] for n in recent],
            [f"{n['night']} · {n['meteors']} 颗流星, {n['rejects']} 个排除" for n in recent],
            w=900, h=200, y_label="每晚流星数") + "</div>")
    body.append("<h2>观测夜</h2>")
    body.append('<div class="tablewrap"><table><thead><tr><th>日期</th><th>流星</th><th>排除</th><th>主要流星雨</th>'
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" if rows
                else '<div class="empty">还没有数据</div>')
    body.append("""<script>
async function s(){try{const r=await fetch('status.json?'+Date.now());const j=await r.json();
const ago=Math.round(Date.now()/1000-j.updated);const t=(l,v,d)=>`<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
document.getElementById('status').innerHTML=
t('运行模式',j.mode==='night'?'夜间监测':'白天待机',ago<120?`${ago} 秒前更新`:`⚠ ${Math.round(ago/60)} 分钟未更新`)+
t('帧率',j.fps??'—',`丢帧 ${j.dropped??0}`)+t('今晚事件',j.events??0,j.last_event??'')+
t('CPU 温度',j.cpu_temp!=null?j.cpu_temp.toFixed(1)+' °C':'—',`增益 ${j.gain??'—'}`)+
t('剩余空间',j.free_gb!=null?j.free_gb+' GB':'—',`太阳高度 ${j.sun_alt??'—'}°`);}catch(e){}}
s();setInterval(s,30000);</script>""")
    (paths.root / "index.html").write_text(page(f"{cfg.site.name} 流星监测", "".join(body)), encoding="utf-8")
