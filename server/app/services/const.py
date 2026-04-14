import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 无论从何处启动，都从 server/.env 加载（与 start-backend 脚本一致）
# override=True：若系统/IDE 里存在空的环境变量，仍允许 .env 里的值生效
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

# 本地 ComfyUI（与 public/comfyui/workflow1.json 配套）
comfyui_base_url = (os.getenv("COMFYUI_BASE_URL") or "http://127.0.0.1:8188").strip().rstrip("/")
comfyui_default_checkpoint = (os.getenv("COMFYUI_DEFAULT_CHECKPOINT") or "").strip()
comfyui_default_workflow_json = (os.getenv("COMFYUI_WORKFLOW_JSON") or "workflow1.json").strip() or "workflow1.json"
# 轮询 /history 直到出现可下载图片的最长等待（秒）；重型工作流（高清修复、多节点）常需 15–30+ 分钟
try:
    comfyui_poll_timeout = float((os.getenv("COMFYUI_POLL_TIMEOUT") or "1800").strip())
except ValueError:
    comfyui_poll_timeout = 1800.0
comfyui_poll_timeout = max(120.0, min(comfyui_poll_timeout, 7200.0))

# 生图分辨率预设 ratio 键（见 comfyui_size_presets）；空或非法时由解析函数回退 1.0
comfyui_default_size_ratio = (os.getenv("COMFYUI_SIZE_RATIO") or "1.0").strip() or "1.0"

# 立绘去背景：相对 public/comfyui/ 的 API 工作流 JSON（需 ComfyUI 安装 RMBG 及模型，如 RMBG-2.0）
comfyui_remove_bg_workflow = (
    (os.getenv("COMFYUI_REMOVE_BG_WORKFLOW") or "sp_costum_workflow/remove_bg.json")
    .strip()
    .replace("\\", "/")
    or "sp_costum_workflow/remove_bg.json"
)
try:
    comfyui_remove_bg_poll_timeout = float((os.getenv("COMFYUI_REMOVE_BG_POLL_TIMEOUT") or "600").strip())
except ValueError:
    comfyui_remove_bg_poll_timeout = 600.0
comfyui_remove_bg_poll_timeout = max(60.0, min(comfyui_remove_bg_poll_timeout, 3600.0))

time_now = datetime.now()
time_now_ = str(time_now).split(".")[0].replace(":", "_")