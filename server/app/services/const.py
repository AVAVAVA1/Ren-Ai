import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 无论从何处启动，都从 server 目录加载 .env（与 start-backend 脚本一致）
_SERVER_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_SERVER_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# DeepSeek 官方兼容 OpenAI 协议；也可用 LLM_API_BASE 指向中转/代理地址
llm_base_url = (os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").strip()
llm_model = (os.getenv("LLM_MODEL") or "deepseek-chat").strip() or "deepseek-chat"

# 优先 DS_API_KEY，与文档一致；兼容 LLM_API_KEY / OPENAI_API_KEY
api_key = (
    os.getenv("DS_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
ds_api_key = os.getenv("DS_API_KEY") or api_key or None
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
modelscope_api_key = os.getenv("MODELSCOPE_API_KEY")
time_now = datetime.now()
time_now_ = str(time_now).split(".")[0].replace(":", "_")