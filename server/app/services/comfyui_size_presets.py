"""
ComfyUI 生图分辨率预设（与常见 SDXL 纵横比表一致）。
ratio 为字符串键，与设置页、请求体一致。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# (ratio, width, height) — 顺序即下拉展示顺序
_SIZE_ROWS: List[Tuple[str, int, int]] = [
    ("0.5", 704, 1408),
    ("0.52", 704, 1344),
    ("0.57", 768, 1344),
    ("0.6", 768, 1280),
    ("0.68", 832, 1216),
    ("0.72", 832, 1152),
    ("0.78", 896, 1152),
    ("0.82", 896, 1088),
    ("0.88", 960, 1088),
    ("0.94", 960, 1024),
    ("1.0", 1024, 1024),
    ("1.07", 1024, 960),
    ("1.13", 1088, 960),
    ("1.21", 1088, 896),
    ("1.29", 1152, 896),
    ("1.38", 1152, 832),
    ("1.46", 1216, 832),
    ("1.67", 1280, 768),
    ("1.75", 1344, 768),
    ("1.91", 1344, 704),
    ("2.0", 1408, 704),
    ("2.09", 1472, 704),
    ("2.4", 1536, 640),
    ("2.5", 1600, 640),
    ("2.89", 1664, 576),
    ("3.0", 1728, 576),
]

SIZE_PRESETS: List[Dict[str, Any]] = [
    {"ratio": r, "width": w, "height": h, "label": f"{r} — {w}×{h}"} for r, w, h in _SIZE_ROWS
]

_RATIO_TO_WH: Dict[str, Tuple[int, int]] = {r: (w, h) for r, w, h in _SIZE_ROWS}


def resolve_size_px(ratio: str) -> Tuple[int, int]:
    """未知 ratio 时回退 1024×1024（1.0）。"""
    key = (ratio or "").strip()
    return _RATIO_TO_WH.get(key, (1024, 1024))


def is_valid_ratio(ratio: str) -> bool:
    return (ratio or "").strip() in _RATIO_TO_WH
