import asyncio
import copy
import os
from datetime import datetime
from typing import Any, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import const, tools
from app.services.get_bg import get_bg
from app.services.comfyui_size_presets import SIZE_PRESETS
from app.services.comfyui_workflow_mapping import list_comfyui_workflows, safe_workflow_json_path
from app.services.get_comfyui_pic import get_comfy_char_pics, comfy_generate_flow_backgrounds
from app.services.get_runninghub_pic import (
    default_runninghub_pic_dir,
    get_running_pic,
    sanitize_character_name_for_path,
)
from app.services.remove_bg import replace_character_stand_pics_with_removed_bg

# 中性前缀：立绘/背景支持 RunningHub 与本地 ComfyUI（access log 不再误显示为仅 RunningHub）
router = APIRouter(prefix="/api/image", tags=["image"])


class GenerateCharacterPicsBody(BaseModel):
    workflow_id: str = Field(
        default="",
        description="RunningHub 工作流 ID；image_backend=runninghub 时必填",
    )
    character_name: str = Field("", description="角色名，用于 public/sources/pic/{name}/ 子目录")
    appearance: str = Field("", description="外貌描述")
    personality: str = Field("", description="性格设定")
    image_backend: str = Field(
        default="runninghub",
        description="生图后端：runninghub 或 comfyui",
    )
    comfyui_checkpoint: str = Field(
        default="",
        description="ComfyUI 模型文件名（与 Web 中 Checkpoint 列表一致）；空则用默认",
    )
    comfyui_workflow: str = Field(
        default="",
        description="public/comfyui 下的工作流 JSON 文件名，如 workflow1.json；空则用服务端默认",
    )
    comfyui_size_ratio: str = Field(
        default="",
        description="分辨率预设 ratio 键（如 1.0、0.5）；空则用服务端默认；仅当 mapping 含 size 时生效",
    )


class RemoveStandPicBgBody(BaseModel):
    character_name: str = Field("", description="与立绘目录一致的角色名（同生成立绘时的名称规则）")


class GenerateFlowBackgroundsBody(BaseModel):
    structured_json: Union[List[Any], dict] = Field(
        ...,
        description="与流程图「导出 JSON」相同结构：dialogue_name、site_description、dialogue_content[]",
    )
    workflow_id: str = Field(
        default="2037179226444533762",
        description="RunningHub 背景工作流 ID（与 get_bg 中节点 11/12 配套）；comfyui 时可忽略",
    )
    image_backend: str = Field(
        default="runninghub",
        description="生图后端：runninghub 或 comfyui",
    )
    comfyui_checkpoint: str = Field(
        default="",
        description="ComfyUI ckpt 文件名；空则用默认",
    )
    comfyui_workflow: str = Field(
        default="",
        description="public/comfyui 下的工作流 JSON；空则用服务端默认",
    )
    comfyui_size_ratio: str = Field(
        default="",
        description="分辨率 ratio 预设；空则用服务端默认",
    )


@router.get("/comfyui-size-presets")
async def comfyui_size_presets():
    """生图宽高预设列表（与设置页下拉一致）。"""
    return {"presets": SIZE_PRESETS, "default_ratio": const.comfyui_default_size_ratio}


@router.get("/comfyui-workflows")
async def comfyui_workflows_list():
    """列出 public/comfyui 中可用的 *.json 工作流及映射来源（.mapping.json / .txt / 内置）。"""
    try:
        items = list_comfyui_workflows()
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"workflows": items, "default_file": const.comfyui_default_workflow_json}


@router.post("/generate-character-pics")
async def generate_character_pics(body: GenerateCharacterPicsBody):
    """
    根据人物外貌、性格调用 get_running_pic（内含 LLM 转提示词与 RunningHub 多表情出图）。
    图片保存目录：public/sources/pic/{角色名}/，文件名为各表情如 happy_1.png（前端可通过 /sources/pic/ 访问）。
    注意：任务含多次排队与轮询，可能耗时数分钟，请保持请求超时足够长。
    """
    backend = (body.image_backend or "runninghub").strip().lower()
    wf = (body.workflow_id or "").strip()
    if backend == "runninghub" and not wf:
        raise HTTPException(status_code=400, detail="使用 RunningHub 时 workflow_id 不能为空")

    parts = []
    if (body.appearance or "").strip():
        parts.append(f"外貌描述：{body.appearance.strip()}")
    if (body.personality or "").strip():
        parts.append(f"性格设定：{body.personality.strip()}")
    positive_prompt = "\n".join(parts).strip()
    if not positive_prompt:
        raise HTTPException(status_code=400, detail="请至少填写外貌描述或性格设定")

    base_pic = default_runninghub_pic_dir()
    char_name = (body.character_name or "").strip()
    folder = sanitize_character_name_for_path(char_name)
    save_dir_out = os.path.join(base_pic, folder)

    ckpt = (body.comfyui_checkpoint or "").strip() or None
    comfy_wf = (body.comfyui_workflow or "").strip() or None
    size_ratio = (body.comfyui_size_ratio or "").strip() or None
    if backend == "comfyui" and comfy_wf:
        try:
            safe_workflow_json_path(comfy_wf)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=f"无效的 ComfyUI 工作流文件: {e}") from e

    def _run():
        if backend == "comfyui":
            get_comfy_char_pics(
                positive_prompt=positive_prompt,
                character_name=char_name,
                save_dir=base_pic,
                checkpoint=ckpt,
                workflow_json=comfy_wf,
                size_ratio=size_ratio,
            )
        else:
            get_running_pic(
                workflowId=wf,
                positive_prompt=positive_prompt,
                character_name=char_name,
                save_dir=base_pic,
            )

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {e!s}") from e

    label = "ComfyUI" if backend == "comfyui" else "RunningHub"
    return {
        "success": True,
        "message": f"已完成 {label} 出图流程，文件已保存到 public/sources/pic/{folder}/",
        "save_dir": save_dir_out,
        "character_folder": folder,
        "public_base": f"/sources/pic/{folder}/",
        "image_backend": backend,
    }


@router.post("/generate-flow-backgrounds")
async def generate_flow_backgrounds(body: GenerateFlowBackgroundsBody):
    """
    按区块 site_description 调用 get_bg 生成背景，保存到 public/pic_bg/{时间戳}/地点{i}.png；
    将该块 dialogue_content 中每条节点的 background 设为 /pic_bg/{时间戳}/地点{i}.png（Vite 静态路径）。
    仅处理 site_description 非空的区块；i 为区块在数组中的序号（从 1 起）。
    """
    backend = (body.image_backend or "runninghub").strip().lower()
    wf = (body.workflow_id or "").strip() or "2037179226444533762"
    ckpt = (body.comfyui_checkpoint or "").strip() or None
    comfy_wf = (body.comfyui_workflow or "").strip() or None
    size_ratio = (body.comfyui_size_ratio or "").strip() or None
    if backend == "comfyui" and comfy_wf:
        try:
            safe_workflow_json_path(comfy_wf)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=f"无效的 ComfyUI 工作流文件: {e}") from e

    raw = body.structured_json
    input_was_single = not isinstance(raw, list)
    blocks: List[dict] = copy.deepcopy(raw if isinstance(raw, list) else [raw])

    time_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_abs = str(tools.get_project_root() / "public" / "pic_bg" / time_folder)
    generated = 0

    def _run():
        nonlocal generated
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            desc = (block.get("site_description") or "").strip()
            if not desc:
                continue
            save_filename = f"地点{idx + 1}.png"
            if backend == "comfyui":
                comfy_generate_flow_backgrounds(
                    [desc],
                    save_dir_abs,
                    [save_filename],
                    checkpoint=ckpt,
                    workflow_json=comfy_wf,
                    size_ratio=size_ratio,
                )
            else:
                get_bg(wf, [desc], save_dir_abs, [save_filename])
            rel = f"/pic_bg/{time_folder}/{save_filename}"
            for node in block.get("dialogue_content") or []:
                if isinstance(node, dict):
                    node["background"] = rel
            generated += 1

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成背景失败: {e!s}") from e

    out: Union[List[dict], dict] = blocks[0] if input_was_single and len(blocks) == 1 else blocks
    return {
        "success": True,
        "dialogues": out,
        "time_folder": time_folder,
        "save_dir": save_dir_abs.replace("\\", "/"),
        "public_base": f"/pic_bg/{time_folder}/",
        "generated_blocks": generated,
        "message": f"已为 {generated} 个有场景描述的区块生成背景并写回节点 background 字段。",
    }


@router.post("/remove-stand-pic-backgrounds")
async def remove_stand_pic_backgrounds(body: RemoveStandPicBgBody):
    """
    对 public/sources/pic/{角色名}/ 下已有表情 PNG 调用 remove.bg 去背景，覆盖原文件；
    返回新的静态 URL 列表，供前端替换该角色的立绘图。
    """
    char_name = (body.character_name or "").strip()
    folder = sanitize_character_name_for_path(char_name)

    def _run():
        return replace_character_stand_pics_with_removed_bg(char_name)

    try:
        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, _run)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"去背景失败: {e!s}") from e

    return {
        "success": True,
        "message": f"已对 {len(images)} 张立绘去背景并写回 public/sources/pic/{folder}/",
        "character_folder": folder,
        "images": images,
    }
