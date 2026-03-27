import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 无论从何处启动，都从 server/.env 加载（与 start-backend 脚本一致）
# override=True：若系统/IDE 里存在空的 REMOVE_BG_API_KEY 等变量，仍允许 .env 里的值生效
_SERVER_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_SERVER_DIR / ".env", override=True)

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
running_hub_key = os.getenv('RunningHub_API_KEY')
# remove.bg：https://www.remove.bg/api — 勿将密钥写入代码，使用 .env
remove_bg_key = (os.getenv("REMOVE_BG_API_KEY") or os.getenv("REMOVE_BG_KEY") or "").strip()
time_now = datetime.now()
time_now_ = str(time_now).split(".")[0].replace(":", "_")