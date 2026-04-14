"""
使用本地 ComfyUI 出图，与 RunningHub 管线对齐：
- 立绘：public/sources/pic/{角色名}/{expression}_1.png 与同目录 {expression}.txt（默认写 expression，自定义写 description）
- 流程背景：public/pic_bg/{时间戳}/地点{i}.png，提示词逻辑同 get_bg（每条末尾追加「不要出现人物」）

工作流 JSON 与注入槽位由 public/comfyui/*.mapping.json 或 *.txt 描述（见 comfyui_workflow_mapping）。
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from app.services import const
from app.services.comfyui_client import ComfyUiClient
from app.services.comfyui_size_presets import resolve_size_px
from app.services.comfyui_workflow_mapping import (
    ComfyWorkflowMapping,
    apply_mapping_to_workflow,
    apply_workflow_dimensions,
    load_workflow_mapping_for_json,
    read_checkpoint_default_from_workflow,
    safe_workflow_json_path,
)
from app.services.get_bg import _append_no_people_to_prompts
from app.services.get_runninghub_pic import (
    EXPRESSION_LS,
    default_runninghub_pic_dir,
    get_art_prompt,
    sanitize_character_name_for_path,
    stand_pic_save_filename,
    write_stand_pic_description_txt,
)

# 与 workflow1 默认负向一致；可被 run_comfyui_save_one_image 覆盖
def _randomize_integer_seeds_in_workflow(workflow: dict) -> None:
    """各次 API 提交使用不同随机种子；仅改写 inputs.seed 为 int 的项（跳过节点连线 [node, slot]）。"""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inp = node.get("inputs")
        if not isinstance(inp, dict):
            continue
        if "seed" in inp and type(inp["seed"]) is int:
            inp["seed"] = random.randrange(0, 2**31)


DEFAULT_COMFYUI_NEGATIVE = (
    "(worst quality, bad quality:1.2),low quality,simple background,transparent,logo,text,"
    "jpeg artifacts,bad anatomy,old,early,copyright name,watermark,artist name,signature,"
    "bad hands,bad fingers,child, loli, symbol-shaped pupils,lips,pov"
)


def _resolved_workflow_path(workflow_json: Optional[str]) -> Path:
    name = (workflow_json or "").strip() or const.comfyui_default_workflow_json
    return safe_workflow_json_path(name)


def load_comfyui_workflow_and_mapping(
    workflow_json: Optional[str] = None,
) -> tuple[dict, ComfyWorkflowMapping]:
    path = _resolved_workflow_path(workflow_json)
    with open(path, encoding="utf-8") as f:
        workflow = json.load(f)
    workflow = copy.deepcopy(workflow)
    _randomize_integer_seeds_in_workflow(workflow)
    mapping = load_workflow_mapping_for_json(path)
    return workflow, mapping


def resolve_comfyui_checkpoint(
    workflow: dict,
    mapping: ComfyWorkflowMapping,
    checkpoint_override: Optional[str],
) -> str:
    o = (checkpoint_override or "").strip()
    if o:
        return o
    env_ckpt = (const.comfyui_default_checkpoint or "").strip()
    if env_ckpt:
        return env_ckpt
    d = read_checkpoint_default_from_workflow(workflow, mapping)
    if d:
        return d
    return "waiIllustriousSDXL_v160 (1).safetensors"


def run_comfyui_save_one_image(
    positive_prompt: str,
    save_path: str,
    *,
    checkpoint: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    base_url: Optional[str] = None,
    workflow_json: Optional[str] = None,
    size_ratio: Optional[str] = None,
) -> str:
    workflow, mapping = load_comfyui_workflow_and_mapping(workflow_json)
    ckpt = resolve_comfyui_checkpoint(workflow, mapping, checkpoint)
    neg = negative_prompt if negative_prompt is not None else DEFAULT_COMFYUI_NEGATIVE
    apply_mapping_to_workflow(
        workflow,
        mapping,
        positive_prompt=positive_prompt,
        negative_prompt=neg,
        checkpoint=ckpt,
    )
    ratio = (size_ratio or "").strip() or const.comfyui_default_size_ratio
    w, h = resolve_size_px(ratio)
    apply_workflow_dimensions(workflow, mapping.size, w, h)
    client = ComfyUiClient(base_url or const.comfyui_base_url)
    images = client.process_workflow(workflow)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    img = images[0]
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(save_path, "PNG")
    return save_path


def get_comfy_char_pics(
    positive_prompt: str,
    character_name: str = "",
    save_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    workflow_json: Optional[str] = None,
    size_ratio: Optional[str] = None,
    stand_items: Optional[List[Tuple[str, str]]] = None,
    image_prompt_extra: Optional[str] = None,
    stand_expression_mode: str = "default",
) -> None:
    """与 get_running_pic 相同目录与文件名规则；stand_items 为 (id, description)，未传则使用 EXPRESSION_LS；每条出图后写同目录描述 .txt。"""
    base_dir = save_dir if save_dir else default_runninghub_pic_dir()
    safe_char = sanitize_character_name_for_path(character_name)
    save_dir_final = os.path.join(base_dir, safe_char)
    os.makedirs(save_dir_final, exist_ok=True)

    items: List[Tuple[str, str]] = (
        list(stand_items) if stand_items else [(e, e) for e in EXPRESSION_LS]
    )
    if not items:
        raise ValueError("立绘列表为空")

    ex = get_art_prompt(positive_prompt)
    extra = (image_prompt_extra or "").strip()
    for stand_id, desc in items:
        sid = (stand_id or "").strip()
        d = (desc or "").strip()
        if not sid or not d:
            raise ValueError("每项立绘须包含非空的 id 与 description")
        pm = f"{ex}, cowboy_shot, {d}"
        if extra:
            pm = f"{pm}, {extra}"
        out_path = os.path.join(save_dir_final, stand_pic_save_filename(sid))
        run_comfyui_save_one_image(
            pm,
            out_path,
            checkpoint=checkpoint,
            base_url=base_url,
            workflow_json=workflow_json,
            size_ratio=size_ratio,
        )
        write_stand_pic_description_txt(
            save_dir_final,
            sid,
            d,
            stand_expression_mode=stand_expression_mode,
        )
        # 给 ComfyUI 队列一点时间收尾，避免极短间隔下一条 history 仍指向上一次任务
        time.sleep(0.35)


def comfy_generate_flow_backgrounds(
    prompts: List[str],
    save_dir_abs: str,
    save_filenames: List[str],
    checkpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    workflow_json: Optional[str] = None,
    size_ratio: Optional[str] = None,
) -> None:
    """与 get_bg 相同的后处理提示词（不要出现人物），逐条出图。"""
    if len(prompts) != len(save_filenames):
        raise ValueError("prompts 与 save_filenames 长度须一致")
    os.makedirs(save_dir_abs, exist_ok=True)
    adjusted = _append_no_people_to_prompts(prompts)
    for text, name in zip(adjusted, save_filenames):
        path = os.path.join(save_dir_abs, name)
        run_comfyui_save_one_image(
            text,
            path,
            checkpoint=checkpoint,
            base_url=base_url,
            workflow_json=workflow_json,
            size_ratio=size_ratio,
        )
        time.sleep(0.35)
