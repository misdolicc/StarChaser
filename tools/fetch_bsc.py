"""下载耶鲁亮星表 (Yale Bright Star Catalog, 5th ed., CDS V/50) 并转换为 starchaser/data/stars.csv。

用法:
    python tools/fetch_bsc.py                 # 从 CDS 下载
    python tools/fetch_bsc.py --file catalog.dat   # 使用本地已下载/解压的 catalog 文件

输出列: hr, name, ra_deg, dec_deg, vmag   (J2000 历元/春分点, 忽略自行; 26 年最大误差约 2 角分)
"""
import argparse
import csv
import gzip
import io
import urllib.request
from pathlib import Path

URL = "https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz"
OUT = Path(__file__).resolve().parents[1] / "starchaser" / "data" / "stars.csv"


def parse(lines, max_mag):
    rows = []
    for line in lines:
        line = line.rstrip("\n")
        if len(line) < 107 or not line[75:77].strip() or not line[102:107].strip():
            continue  # 无 J2000 坐标或无星等的条目 (新星/非恒星)
        hr = int(line[0:4])
        name = " ".join(line[4:14].split())
        ra = (int(line[75:77]) + int(line[77:79]) / 60 + float(line[79:83]) / 3600) * 15
        dec = int(line[84:86]) + int(line[86:88]) / 60 + int(line[88:90]) / 3600
        if line[83] == "-":
            dec = -dec
        vmag = float(line[102:107])
        if vmag <= max_mag:
            rows.append((hr, name, round(ra, 5), round(dec, 5), vmag))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="本地 catalog 文件 (可为 .gz)")
    ap.add_argument("--max-mag", type=float, default=6.0)
    args = ap.parse_args()
    if args.file:
        raw = Path(args.file).read_bytes()
    else:
        print(f"downloading {URL} ...")
        raw = urllib.request.urlopen(URL, timeout=60).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    rows = parse(io.StringIO(raw.decode("ascii", errors="replace")), args.max_mag)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hr", "name", "ra_deg", "dec_deg", "vmag"])
        w.writerows(rows)
    print(f"wrote {len(rows)} stars (V <= {args.max_mag}) -> {OUT}")


if __name__ == "__main__":
    main()
