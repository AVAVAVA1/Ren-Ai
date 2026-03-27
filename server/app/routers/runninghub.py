import asyncio
import copy
import os
from datetime import datetime
from typing import Any, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import tools
from app.services.get_bg import get_bg
from app.services.get_runninghub_pic import (
    default_runninghub_pic_dir,
    get_running_pic,
    sanitize_character_name_for_path,
)
from app.services.remove_bg import replace_character_stand_pics_with_removed_bg

router = APIRouter(prefix="/api/runninghub", tags=["runninghub"])


class GenerateCharacterPicsBody(BaseModel):
    workflow_id: str = Field(..., description="RunningHub 工作流 ID，与设置页一致")
    character_name: str = Field("", description="角色名，用于 public/sources/pic/{name}/ 子目录")
    appearance: str = Field("", description="外貌描述")
    personality: str = Field("", description="性格设定")


class RemoveStandPicBgBody(BaseModel):
    character_name: str = Field("", description="与立绘目录一致的角色名（同生成立绘时的名称规则）")


class GenerateFlowBackgroundsBody(BaseModel):
    structured_json: Union[List[Any], dict] = Field(
        ...,
        description="与流程图「导出 JSON」相同结构：dialogue_name、site_description、dialogue_content[]",
    )
    workflow_id: str = Field(
        default="2037179226444533762",
        description="RunningHub 背景工作流 ID（与 get_bg 中节点 11/12 配套）",
    )


@router.post("/generate-character-pics")
async def generate_character_pics(body: GenerateCharacterPicsBody):
    """
    根据人物外貌、性格调用 get_running_pic（内含 LLM 转提示词与 RunningHub 多表情出图）。
    图片保存目录：public/sources/pic/{角色名}/，文件名为各表情如 happy_1.png（前端可通过 /sources/pic/ 访问）。
    注意：任务含多次排队与轮询，可能耗时数分钟，请保持请求超时足够长。
    """
    wf = (body.workflow_id or "").strip()
    if not wf:
        raise HTTPException(status_code=400, detail="workflow_id 不能为空")

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

    def _run():
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

    return {
        "success": True,
        "message": f"已完成 RunningHub 出图流程，文件已保存到 public/sources/pic/{folder}/",
        "save_dir": save_dir_out,
        "character_folder": folder,
        "public_base": f"/sources/pic/{folder}/",
    }


@router.post("/generate-flow-backgrounds")
async def generate_flow_backgrounds(body: GenerateFlowBackgroundsBody):
    """
    按区块 site_description 调用 get_bg 生成背景，保存到 public/pic_bg/{时间戳}/地点{i}.png；
    将该块 dialogue_content 中每条节点的 background 设为 /pic_bg/{时间戳}/地点{i}.png（Vite 静态路径）。
    仅处理 site_description 非空的区块；i 为区块在数组中的序号（从 1 起）。
    """
    wf = (body.workflow_id or "").strip() or "2037179226444533762"
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
