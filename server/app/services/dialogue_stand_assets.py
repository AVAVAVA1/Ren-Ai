"""故事对话阶段：已选角色立绘可选标识（与 public/sources/pic 下 .txt 配套）及通用精灵图路径。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from app.services.get_runninghub_pic import (
    default_runninghub_pic_dir,
    list_stand_pic_png_paths,
    sanitize_character_name_for_path,
)

# LLM 对非插入名单说话者填写的占位符，后处理替换为通用精灵站点路径
SPIRIT_CHARACTER_TOKEN = "__SPIRIT_DEFAULT__"
SPIRIT_PUBLIC_PATH = "/sources/spirit/ComfyUI_00016_black.png"
# 旁白无立绘：后处理为 character 空字符串
NARRATION_CHARACTER_TOKEN = "__NARRATION_EMPTY__"


def normalize_story_cast(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """character_name → 与立绘目录一致的 folder（sanitize_character_name_for_path）。"""
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("character_name") or item.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "folder": sanitize_character_name_for_path(name)})
    return out


def list_txt_stem_descriptions(pic_dir: str) -> List[Tuple[str, str]]:
    """返回 (txt 无后缀文件名, 文件内容) 列表，按 stem 排序。"""
    if not pic_dir or not os.path.isdir(pic_dir):
        return []
    rows: List[Tuple[str, str]] = []
    try:
        names = os.listdir(pic_dir)
    except OSError:
        return []
    for fn in names:
        if not fn.lower().endswith(".txt"):
            continue
        stem, _ = os.path.splitext(fn)
        stem = stem.strip()
        if not stem:
            continue
        path = os.path.join(pic_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                body = f.read().strip()
        except OSError:
            body = ""
        rows.append((stem, body))
    rows.sort(key=lambda x: x[0].lower())
    return rows


def public_sprite_path_for_stem(folder: str, stem: str) -> Optional[str]:
    """将立绘 stem 解析为站点路径 /sources/pic/{folder}/{实际 png 文件名}。"""
    base = default_runninghub_pic_dir()
    pic_dir = os.path.join(base, folder)
    candidates = list_stand_pic_png_paths(pic_dir, stem)
    if not candidates:
        return None
    basename = os.path.basename(candidates[0])
    folder_slash = folder.replace("\\", "/")
    return f"/sources/pic/{folder_slash}/{basename}"


def first_sprite_public_path_in_folder(folder: str) -> Optional[str]:
    """该角色目录下按 txt stem 顺序取第一张可用立绘；无素材则 None。"""
    base = default_runninghub_pic_dir()
    pic_dir = os.path.join(base, folder)
    for stem, _ in list_txt_stem_descriptions(pic_dir):
        p = public_sprite_path_for_stem(folder, stem)
        if p:
            return p
    return None


def resolve_cast_sprite_public_path(folder: str, stem: str) -> Optional[str]:
    """
    将 LLM 给出的 stem 映射到立绘路径：精确匹配 png → 与目录内 txt stem 大小写无关匹配
    → 仍无时用该角色目录下第一个可用立绘（避免 Larok 仅有 happy/sad/sexy 时模型误选 smile 落到精灵图）。
    """
    s = (stem or "").strip()
    if not s:
        return first_sprite_public_path_in_folder(folder)
    hit = public_sprite_path_for_stem(folder, s)
    if hit:
        return hit
    base = default_runninghub_pic_dir()
    pic_dir = os.path.join(base, folder)
    for st, _ in list_txt_stem_descriptions(pic_dir):
        if st.casefold() == s.casefold():
            hit2 = public_sprite_path_for_stem(folder, st)
            if hit2:
                return hit2
    return first_sprite_public_path_in_folder(folder)


def rewrite_dialogue_character_to_paths(
    dialogues: List[Dict[str, Any]],
    cast_entries: List[Dict[str, str]],
) -> None:
    """
    将 LLM 输出的 character 转为流程/pygame 可用值：
    - 旁白 → 空字符串（不立绘）；
    - 非名单角色 → SPIRIT_PUBLIC_PATH；
    - 已选角色 → stem 解析为 /sources/pic/...；非法 stem 回退该角色首张立绘；仅目录无素材时用 SPIRIT。
    """
    by_name = {e["name"]: e for e in cast_entries}
    for row in dialogues:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        raw_ch = (row.get("character") or "").strip()
        if name == "旁白":
            row["character"] = ""
            continue
        if name not in by_name:
            row["character"] = SPIRIT_PUBLIC_PATH
            continue
        folder = by_name[name]["folder"]
        if raw_ch == SPIRIT_CHARACTER_TOKEN or not raw_ch:
            fb = first_sprite_public_path_in_folder(folder)
            row["character"] = fb if fb else SPIRIT_PUBLIC_PATH
            continue
        resolved = resolve_cast_sprite_public_path(folder, raw_ch)
        row["character"] = resolved if resolved else SPIRIT_PUBLIC_PATH


def build_cast_character_instruction_block(cast_entries: List[Dict[str, str]]) -> str:
    """注入提示词：每名已选角色的可选 stem + txt 摘要；旁白与名单外占位符不同。"""
    base = default_runninghub_pic_dir()
    lines: List[str] = [
        "【立绘标识 character 规则（须严格遵守）】",
        "【各角色列表互斥】每名角色对应磁盘上独立子目录，下列「可选标识」按人分开列出、彼此不同。"
        "仅当本条对话的 name 与某角色名完全一致时，character 才可从该角色名下所列标识中择一；"
        "禁止把角色 A 下列出的标识用于角色 B 的台词，禁止借用其他角色目录里才有的标识（即使语义相近也不允许）。",
        f"1）说话者 name 为「旁白」：character 字段必须恰好填写：{NARRATION_CHARACTER_TOKEN}（表示无立绘、勿填路径或表情词）。",
        f"2）说话者 name 不在下方「已选角色」名单内（且不是旁白）：character 必须恰好填写：{SPIRIT_CHARACTER_TOKEN}。",
        "3）说话者 name 等于下方某一「角色名」时：character 必须且只能从该角色名下缩进列出的标识中选一项字符串（与之一字不差），"
        "不得选用同文件其他角色块中的标识，不得自造未列出的词。",
        "",
        "—— 以下为各角色专有可选标识（每块仅适用于 name 与该角色名相同的对话行）——",
    ]
    for e in cast_entries:
        pic_dir = os.path.join(base, e["folder"])
        opts = list_txt_stem_descriptions(pic_dir)
        lines.append(f"·【仅 name=「{e['name']}」】立绘目录专属标识（其他角色的台词禁止使用本块任一标识）：")
        if not opts:
            lines.append(
                f"  - 该目录无 .txt 立绘说明，本角色 character 仅可填：{SPIRIT_CHARACTER_TOKEN}"
            )
            continue
        allowed = "、".join(f"「{st}」" for st, _ in opts)
        lines.append(f"  - 本角色允许的 character 取值集合（仅此集合）：{allowed}")
        for stem, desc in opts:
            short = desc.replace("\n", " ").strip()
            if len(short) > 220:
                short = short[:220] + "…"
            lines.append(f"  - 标识「{stem}」：{short or '（无描述）'}")
    lines.append("")
    lines.append(
        f"再次强调：旁白填 {NARRATION_CHARACTER_TOKEN}；非名单角色（非旁白）填 {SPIRIT_CHARACTER_TOKEN}；"
        "已选角色每条台词的 character 必须落在该 name 对应块所列集合内，禁止跨角色混用。"
    )
    return "\n".join(lines)
