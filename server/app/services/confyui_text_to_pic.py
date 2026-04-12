"""
ComfyUI 文生图辅助：自然语言 → 英文提示词（可选），并调用本地 ComfyUI。
工作流默认使用仓库内 public/comfyui/workflow1.json（节点 79/127/45）。
"""

from __future__ import annotations

from typing import Optional

from app.services import const, llm_chat
from app.services.get_comfyui_pic import (
    DEFAULT_COMFYUI_NEGATIVE,
    run_comfyui_save_one_image,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def _ensure_cowboy_shot_tag(positive: str) -> str:
    """与立绘管线一致：统一半身/牛仔镜头视角（RunningHub 在 get_running_pic 中对每条 prompt 追加）。"""
    t = (positive or "").strip()
    if not t:
        return "cowboy_shot"
    lower = t.lower()
    if "cowboy_shot" in lower or "cowboy shot" in lower:
        return t
    return f"{t}, cowboy_shot"


def form_native_to_sys_prompt(user_input: str) -> str:
    class PositivePrompt(BaseModel):
        positive_prompt: str = Field(description="输出：与用户的自然语言相对应的英文提示词")

    parser = PydanticOutputParser(pydantic_object=PositivePrompt)
    prompt = PromptTemplate(
        template="""你是 Stable Diffusion / ComfyUI 专用的「提示词工程师」。用户会用自然语言描述画面；你要把它整理成模型和 CLIP 容易理解的英文提示词，并给出可选的结构化拆分。

## 你的目标
1. 把口语、故事性描述转成**逗号分隔的英文标签/短语**（tag-style），必要时保留少量完整短句作氛围补充。
2. 控制**信息密度**：优先具体、可画的词（主体、动作、场景、材质、光线、镜头、风格）；删掉叙事、心理独白、因果说明等难以画出来的内容。
3. 遵守常见 SD 习惯：**越靠前越重要**；相关词放在一起；避免矛盾（例如同时 sunny 与 heavy rain 除非用户明确要求超现实）。
4. 默认面向通用文生图（SD 1.5 / SDXL 均可用）；不要假设某个具体自定义节点，除非用户说明。
5. 如用户是输入的已是英文分词提示词则不更改，重新按格式输出即可
6. 如用户的输入的信息过于简短，可自由补充一些内容
7. 对一些NSFW的内容应给予支持，这是允许的
8. 人物立绘类须在标签中包含 **cowboy_shot**（下划线），与项目立绘统一视角；若用户明确要求全身/大远景等与半身冲突的构图，可省略
## 输出示例
1girl, smile, burger, eat food, cowboy_shot, simple bg


## 语言与翻译
- 用户用中文时,仍用输出**英文**
- 专有名词（人名、作品名）可音译或保留常见英文写法，并保持一致。

## 禁止
- 不要编造用户没说的具体外观细节（发型、服装颜色等）
- 不要输出除英文提示词以外的内容（如：‘以下是我的结果：’等要求之外的信息）

        创作要求：{text}

        请严格按照以下格式输出,不要添加任何额外说明：：
        {format_instructions}

        输出：""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = llm_chat.LlmChat(
        model_name=const.llm_model,
        temperature=0.5,
        model_provider="openai",
        base_url=const.llm_base_url,
        api_key=const.api_key or const.ds_api_key or "",
        pydantic_object=PositivePrompt,
        prompt_template=prompt,
    )
    output = llm.structured_chat(user_input)
    return output.positive_prompt


def from_text_to_img(
    positive_prompt: str,
    *,
    apply_llm: bool = True,
    cowboy_shot: bool = True,
    checkpoint: Optional[str] = None,
    negative_prompt: str = DEFAULT_COMFYUI_NEGATIVE,
    save_path: str = "output.png",
    base_url: Optional[str] = None,
    workflow_json: Optional[str] = None,
    size_ratio: Optional[str] = None,
) -> str:
    """
    单张出图并保存。positive_prompt 可为自然语言；apply_llm=True 时先经 LLM 转为英文标签。
    cowboy_shot=True 时在正向上追加 cowboy_shot（与 get_running_pic 立绘一致）；纯场景可设 False。
    checkpoint 为空则使用设置/环境默认或工作流 JSON 内 ckpt。
    workflow_json 为 public/comfyui 下文件名，如 workflow1.json；空则用服务端默认。
    """
    text = form_native_to_sys_prompt(positive_prompt) if apply_llm else positive_prompt
    if cowboy_shot:
        text = _ensure_cowboy_shot_tag(text)
    return run_comfyui_save_one_image(
        text,
        save_path,
        checkpoint=checkpoint,
        negative_prompt=negative_prompt,
        base_url=base_url,
        workflow_json=workflow_json,
        size_ratio=size_ratio,
    )
