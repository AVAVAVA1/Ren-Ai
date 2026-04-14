import copy
import json
import os
from pathlib import Path
from typing import List, Optional

from app.services import const, tools
from app.services.comfyui_client import ComfyUiClient
from app.services.get_runninghub_pic import (
    EXPRESSION_LS,
    default_runninghub_pic_dir,
    list_stand_pic_png_paths,
    sanitize_character_name_for_path,
)


def _resolved_remove_bg_workflow_path() -> Path:
    base = (tools.get_project_root() / "public" / "comfyui").resolve()
    full = (base / const.comfyui_remove_bg_workflow).resolve()
    try:
        full.relative_to(base)
    except ValueError as e:
        raise ValueError("非法的去背景工作流路径") from e
    return full


def remove_bg(file_path: str, output_path: str) -> None:
    """
    使用本地 ComfyUI（public/comfyui 下 RMBG 工作流）去背景，将 PNG 写入 output_path（可含透明通道）。
    依赖 COMFYUI_BASE_URL 可访问，且 ComfyUI 已安装 RMBG 等节点与工作流内模型一致。
    """
    wf_path = _resolved_remove_bg_workflow_path()
    if not wf_path.is_file():
        raise FileNotFoundError(f"未找到去背景工作流: {wf_path}")

    with open(wf_path, encoding="utf-8") as f:
        workflow = copy.deepcopy(json.load(f))

    node1 = workflow.get("1")
    if not isinstance(node1, dict) or not isinstance(node1.get("inputs"), dict):
        raise RuntimeError("去背景工作流缺少节点 1 (LoadImage) 或其 inputs")

    client = ComfyUiClient(const.comfyui_base_url)
    load_key = client.upload_image_to_input(file_path)
    workflow["1"]["inputs"]["image"] = load_key

    images = client.process_workflow(
        workflow,
        poll_timeout=float(const.comfyui_remove_bg_poll_timeout),
    )
    if not images:
        raise RuntimeError("ComfyUI 去背景未返回图片")
    img = images[0]
    img.save(output_path, "PNG")


def replace_character_stand_pics_with_removed_bg(
    character_name: str,
    stand_expression_ids: Optional[List[str]] = None,
) -> List[dict]:
    """
    对 public/sources/pic/{角色名}/ 下立绘 PNG 去背景并覆盖原文件。
    每个表情匹配 happy.png、happy_1.png、happy_2.png 等形式（见 list_stand_pic_png_paths）。
    stand_expression_ids 非空时仅处理这些 id（与自定义立绘文件名一致）；否则处理内置 EXPRESSION_LS。
    """
    folder = sanitize_character_name_for_path((character_name or "").strip())
    base = default_runninghub_pic_dir()
    dir_path = os.path.join(base, folder)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"未找到角色立绘目录: {dir_path}")

    expr_list = (
        [x.strip() for x in stand_expression_ids if x and str(x).strip()]
        if stand_expression_ids
        else list(EXPRESSION_LS)
    )
    if not expr_list:
        expr_list = list(EXPRESSION_LS)

    out: List[dict] = []
    for expr in expr_list:
        for path in list_stand_pic_png_paths(dir_path, expr):
            filename = os.path.basename(path)
            remove_bg(path, path)
            out.append({"url": f"/sources/pic/{folder}/{filename}", "name": filename})

    if not out:
        try:
            existing = sorted(os.listdir(dir_path))
        except OSError:
            existing = []
        expected_sample = "happy.png / happy_1.png、surprise_2.png 等（id + 可选 _数字）"
        hint_name = (
            "当前角色名为空时会使用文件夹「unnamed」；若生成立绘时填过名字，两边必须一致，否则会找错目录。"
        )
        files_hint = f"目录内现有文件（节选）：{existing[:25]}" if existing else "目录内没有任何文件。"
        sample_ids = "、".join(expr_list[:3]) if expr_list else ""
        raise ValueError(
            f"在「{dir_path}」下没有找到可处理的立绘：需要与所选 id 对应的 "
            f"{expected_sample}（示例 id：{sample_ids}）。{hint_name} {files_hint}"
        )
    return out
