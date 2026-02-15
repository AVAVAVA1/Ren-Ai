from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent 状态定义

    这是 Agent 在执行过程中维护的所有数据
    """
    # 必需字段
    messages: List[BaseMessage]  # 对话历史

    # 可选字段（后面会讲）
    user_name: str
    session_id: str

from langchain_core.messages import (
    HumanMessage,    # 用户消息
    AIMessage,       # AI 消息
    SystemMessage,   # 系统消息
    ToolMessage      # 工具返回消息
)

# 消息列表示例  ================================================================================
messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮你？"),
    HumanMessage(content="天气怎么样？"),
    AIMessage(content="让我查询一下..."),
    ToolMessage(content="北京：晴天", tool_call_id="call_123"),
    AIMessage(content="北京今天是晴天")
]