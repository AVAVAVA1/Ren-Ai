import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services import character_chat_llm

router = APIRouter(prefix="/api/character-chat", tags=["character-chat"])


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str = ""


class CharacterChatSendBody(BaseModel):
    """character_name 与人物卡「姓名」一致，用于 public/character_chat 下文件名。"""

    character_name: str = Field("", description="用于文件名的角色名（通常与 card.name 相同）")
    card: Dict[str, Any] = Field(default_factory=dict)
    messages: List[ChatMessageItem] = Field(default_factory=list)


@router.get("/history")
async def get_chat_history(character_name: str = Query("", description="角色姓名")):
    name = (character_name or "").strip() or "未命名"

    def _run():
        return character_chat_llm.load_history_file(name)

    try:
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取对话历史失败: {e!s}") from e
    return {"messages": messages}


@router.post("/send")
async def send_chat_message(body: CharacterChatSendBody):
    name_key = (body.character_name or "").strip() or (str(body.card.get("name") or "").strip()) or "未命名"
    raw_msgs = [{"role": m.role, "content": m.content} for m in body.messages]

    if not raw_msgs or raw_msgs[-1].get("role") != "user":
        raise HTTPException(status_code=400, detail="messages 非空且最后一条须为用户消息")

    def _run():
        try:
            _, full = character_chat_llm.run_chat_round(body.card or {}, raw_msgs)
        except ValueError as e:
            raise ValueError(str(e)) from e
        display = (str(body.card.get("name") or "")).strip() or name_key
        character_chat_llm.save_history_file(name_key, display, full)
        return full

    try:
        loop = asyncio.get_running_loop()
        full = await loop.run_in_executor(None, _run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"模型调用或保存失败: {e!s}") from e

    return {"messages": full, "success": True}
