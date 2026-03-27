import os
from typing import List

import requests

from app.services import const
from app.services.get_runninghub_pic import (
    EXPRESSION_LS,
    default_runninghub_pic_dir,
    list_stand_pic_png_paths,
    sanitize_character_name_for_path,
)


def remove_bg(file_path: str, output_path: str) -> None:
    """
    调用 remove.bg API 去背景，将透明底 PNG 写入 output_path。
    需在环境变量中配置 REMOVE_BG_API_KEY（或 REMOVE_BG_KEY）。
    """
    key = (const.remove_bg_key or "").strip()
    if not key:
        raise ValueError(
            "未配置 remove.bg 密钥：请在 server/.env 中设置 REMOVE_BG_API_KEY（或 REMOVE_BG_KEY）。"
        )

    with open(file_path, "rb") as image_file:
        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": image_file},
            data={"size": "auto"},
            headers={"X-Api-Key": key},
            timeout=120,
        )

    if response.status_code == requests.codes.ok:
        with open(output_path, "wb") as out:
            out.write(response.content)
        return

    raise RuntimeError(
        f"remove.bg 去背景失败: HTTP {response.status_code} {response.text[:800]}"
    )


def replace_character_stand_pics_with_removed_bg(character_name: str) -> List[dict]:
    """
    对 public/sources/pic/{角色名}/ 下立绘 PNG 去背景并覆盖原文件。
    每个表情匹配 happy.png、happy_1.png、happy_2.png 等形式（见 list_stand_pic_png_paths）。
    """
    folder = sanitize_character_name_for_path((character_name or "").strip())
    base = default_runninghub_pic_dir()
    dir_path = os.path.join(base, folder)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"未找到角色立绘目录: {dir_path}")

    out: List[dict] = []
    for expr in EXPRESSION_LS:
        for path in list_stand_pic_png_paths(dir_path, expr):
            filename = os.path.basename(path)
            remove_bg(path, path)
            out.append({"url": f"/sources/pic/{folder}/{filename}", "name": filename})

    if not out:
        try:
            existing = sorted(os.listdir(dir_path))
        except OSError:
            existing = []
        expected_sample = "happy.png / happy_1.png、surprise_2.png 等（表情名 + 可选 _数字）"
        hint_name = (
            "当前角色名为空时会使用文件夹「unnamed」；若生成立绘时填过名字，两边必须一致，否则会找错目录。"
        )
        files_hint = f"目录内现有文件（节选）：{existing[:25]}" if existing else "目录内没有任何文件。"
        raise ValueError(
            f"在「{dir_path}」下没有找到可处理的立绘：需要 {EXPRESSION_LS[0]}、{EXPRESSION_LS[1]} 等表情对应的 "
            f"{expected_sample}。{hint_name} {files_hint}"
        )
    return out
