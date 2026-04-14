"""
将 RGBA 图中所有 alpha > 0 的像素的 RGB 置为纯黑 #000000，alpha 不变。
用法：python rgba_to_black_silhouette.py <输入.png> [输出.png]
  省略输出时，在同目录生成 <原名>_black.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def rgba_to_black_silhouette(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGBA")
    arr = np.asarray(im, dtype=np.uint8).copy()
    opaque = arr[:, :, 3] > 0
    arr[opaque, 0] = 0
    arr[opaque, 1] = 0
    arr[opaque, 2] = 0
    Image.fromarray(arr).save(dst, "PNG")


def main() -> int:
    p = argparse.ArgumentParser(description="非透明区域改为纯黑，保留透明通道")
    p.add_argument("input", type=Path, help="输入图片路径")
    p.add_argument("output", type=Path, nargs="?", help="输出 PNG 路径（默认 原名_black.png）")
    args = p.parse_args()
    src = args.input.expanduser().resolve()
    if not src.is_file():
        print(f"找不到文件: {src}", file=sys.stderr)
        return 1
    if args.output:
        dst = args.output.expanduser().resolve()
    else:
        dst = src.with_name(f"{src.stem}_black{src.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgba_to_black_silhouette(src, dst)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
