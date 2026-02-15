from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
import json
import const
# agent 要使用低temp吗
model = init_chat_model(
    model="deepseek-chat",
    temperature=0.0,
    model_provider='openai',
    base_url='https://api.deepseek.com',
    api_key=const.api_key,

)


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。
       回答一定要带一个颜文字
    """
    return f"{city}总是阳光明媚！QAQ OwO"


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    system_prompt='',
    tools=[get_weather_for_location],
)
# 维护对话历史
messages = []
while True:
    user_input = input("👤 你: ")
    if user_input.lower() == 'quit':
        break
        # 添加用户消息
    messages.append({"role": "user", "content": user_input})

    # 调用 agent
    result = agent.invoke({"messages": messages})

    # 更新消息历史（包含所有中间步骤）
    messages = result["messages"]

    # 获取最后一条 AI 回复
    last_message = messages[-1]
    if last_message.type == "ai":
        print(f"🤖 AI: {last_message.content}\n")
