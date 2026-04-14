"""
人物卡对话：按角色名读写 public/character_chat/{safe_name}.json；
系统提示词注入角色卡；长对话时对较早轮次做摘要压缩后再送入模型。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services import const, tools
from app.services.get_runninghub_pic import sanitize_character_name_for_path
from app.services.llm_chat import message_to_text

# 送入模型的最近消息条数（user/assistant 交替）
CONTEXT_TAIL_MAX = 22
# 超过该条数则对更早内容做一次性摘要并写入 system
COMPRESS_TRIGGER = 26
SUMMARY_INPUT_CAP = 12000
SUMMARY_OUTPUT_CAP = 420


def _chat_dir() -> Path:
    d = tools.get_project_root() / "public" / "character_chat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def history_path(character_name: str) -> Path:
    safe = sanitize_character_name_for_path(character_name)
    return _chat_dir() / f"{safe}.json"


def build_system_prompt(card: Dict[str, Any]) -> str:
    parts: List[str] = [
        "你正在与用户进行沉浸式中文角色扮演对话。请严格保持角色身份与口吻。",
        "重要规则：禁止代用户（User）发言、替用户做决定或描写用户未明确表达的行为；"
        "用户的话只来自下方标记为用户的对话内容。",
        "",
        "【角色设定卡】",
    ]
    name = (str(card.get("name") or "")).strip() or "未命名"
    parts.append(f"姓名：{name}")
    age = (str(card.get("age") or "")).strip()
    if age:
        parts.append(f"年龄：{age}")
    gender = (str(card.get("gender") or "")).strip()
    if gender:
        parts.append(f"性别：{gender}")
    appearance = (str(card.get("appearance") or "")).strip()
    if appearance:
        parts.append(f"外貌：{appearance}")
    personality = (str(card.get("personality") or "")).strip()
    if personality:
        parts.append(f"性格：{personality}")
    background = (str(card.get("background") or "")).strip()
    if background:
        parts.append(f"背景：{background}")
    de = (str(card.get("dialogue_examples") or "")).strip()
    if de:
        parts.append(
            "\n【对话风格示例】（仅供语气与用词参考，勿照搬或复述示例全文）\n" + de
        )
    other = (str(card.get("other_settings") or "")).strip()
    if other:
        parts.append("\n【其他设定】\n" + other)
    parts.append(
        "\n请以该角色的自然口吻回复；回复长度随情境收放；勿输出元说明（如「作为 AI」等）。"
    )
    return "\n".join(parts)


def _make_llm(temperature: float = 0.7):
    key = str(const.api_key or "").strip()
    if not key:
        raise ValueError(
            "未配置大模型 API Key：请在 server/.env 中设置 DS_API_KEY 或 LLM_API_KEY。"
        )
    return init_chat_model(
        model=const.llm_model,
        temperature=temperature,
        model_provider="openai",
        base_url=const.llm_base_url,
        api_key=key,
    )


def summarize_messages(old_slice: List[Dict[str, str]]) -> str:
    """将较早对话轮次压缩为短摘要，用于 system 拼接。"""
    if not old_slice:
        return ""
    lines: List[str] = []
    for m in old_slice:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        tag = "用户" if role == "user" else ("角色" if role == "assistant" else role)
        lines.append(f"{tag}：{content}")
    blob = "\n".join(lines)
    if not blob.strip():
        return ""
    blob = blob[:SUMMARY_INPUT_CAP]
    llm = _make_llm(0.25)
    prompt = (
        "请将以下对话压缩为一段中文摘要，供后续轮次理解前情。要求：\n"
        "1）不超过 380 字；2）保留关键事实、称呼、约定与情绪转折；\n"
        "3）严禁编造对话中未出现的内容。\n\n"
        f"对话内容：\n{blob}"
    )
    r = llm.invoke(prompt)
    text = message_to_text(r).strip()
    return text[:SUMMARY_OUTPUT_CAP] if text else ""


def _normalize_turns(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def build_langchain_messages(
    card: Dict[str, Any], full_messages: List[Dict[str, str]]
) -> List[Any]:
    """构造 invoke 用的消息列表（含 System）；full_messages 须含末尾用户句。"""
    fm = _normalize_turns(full_messages)
    if not fm or fm[-1]["role"] != "user":
        raise ValueError("有效对话末尾须为用户消息")

    system_base = build_system_prompt(card)
    if len(fm) <= COMPRESS_TRIGGER:
        tail = fm[-CONTEXT_TAIL_MAX:] if len(fm) > CONTEXT_TAIL_MAX else fm
        system = system_base
    else:
        old = fm[:-CONTEXT_TAIL_MAX]
        tail = fm[-CONTEXT_TAIL_MAX:]
        summ = summarize_messages(old)
        system = system_base
        if summ:
            system += "\n\n【此前多轮对话压缩摘要】\n" + summ

    out: List[Any] = [SystemMessage(content=system)]
    for m in tail:
        c = m["content"]
        if m["role"] == "user":
            out.append(HumanMessage(content=c))
        else:
            out.append(AIMessage(content=c))
    return out


def load_history_file(character_name: str) -> List[Dict[str, str]]:
    p = history_path(character_name)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(raw, list):
        return _normalize_turns(raw)
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        return _normalize_turns(raw["messages"])
    return []


def save_history_file(
    character_name: str, display_name: str, messages: List[Dict[str, str]]
) -> None:
    _chat_dir()
    p = history_path(character_name)
    doc = {
        "version": 1,
        "character_display_name": (display_name or "").strip() or character_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": _normalize_turns(messages),
    }
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def run_chat_round(
    card: Dict[str, Any], messages_ending_with_user: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, str]]]:
    """
    messages_ending_with_user：完整历史且最后一条为用户。
    返回 (助手回复纯文本, 含助手新句的完整列表)。
    """
    fm = _normalize_turns(messages_ending_with_user)
    if not fm or fm[-1]["role"] != "user":
        raise ValueError("最后一条须为用户消息")

    llm = _make_llm(0.72)
    msgs = build_langchain_messages(card, fm)
    resp = llm.invoke(msgs)
    assistant_text = message_to_text(resp).strip() or "……"

    full = list(fm)
    full.append({"role": "assistant", "content": assistant_text})
    return assistant_text, full
