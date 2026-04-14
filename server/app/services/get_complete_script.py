import re
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services import tools
from app.services.llm_generator import LLMGenerator

_CHARACTER_NAME_LOCK = (
    "【姓名锁定】若用户数据或大纲中给出了角色的正式姓名，则剧本正文里出现的该角色称呼必须与之一字不差；"
    "禁止自行翻译（例如英文变中文）、改名、缩写，或用昵称替代正式姓名。"
)


def complete_script(user_input: str, outline, path: str, strict_model: bool = False):
    class ArticleCompleteScript(BaseModel):
        """剧本结构化输出；校验放宽以兼容模型常犯的格式错误（如 content 误为整段字符串）。"""

        model_config = ConfigDict(extra="ignore")

        article_script_name: str = Field(default="", description="剧本的名字")
        paragraph_num: int = Field(default=0, description="剧本的分段的数目")
        content: List[str] = Field(default_factory=list, description="剧本的完整内容，按段落分")
        site: str = Field(default="", description="主要场景地点概括")

        @field_validator("article_script_name", "site", mode="before")
        @classmethod
        def _str_fields(cls, v):
            if v is None:
                return ""
            return str(v).strip()

        @field_validator("paragraph_num", mode="before")
        @classmethod
        def _paragraph_num(cls, v):
            if v is None:
                return 0
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 0

        @field_validator("content", mode="before")
        @classmethod
        def _content(cls, v):
            if v is None:
                return []
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return []
                parts = [p.strip() for p in re.split(r"\n{2,}", s) if p.strip()]
                return parts if len(parts) > 1 else [s]
            if isinstance(v, list):
                out: List[str] = []
                for item in v:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("body") or item.get("content")
                        if text is not None:
                            out.append(str(text).strip())
                        else:
                            out.append(str(item).strip())
                    else:
                        t = str(item).strip()
                        if t:
                            out.append(t)
                return out
            return []

        @model_validator(mode="after")
        def _sync(self):
            if not self.content:
                raise ValueError("剧本 content 不能为空，请至少输出一段正文")
            object.__setattr__(self, "paragraph_num", len(self.content))
            if not self.article_script_name:
                object.__setattr__(self, "article_script_name", "生成的剧本")
            return self
    
    generator = LLMGenerator(
        pydantic_object=ArticleCompleteScript,
        result_type='剧本'
    )
    
    get_outline = outline
    
    def format_script_input(user_input, previous_result):
        return (
            f"用户要求和待修改的剧本：{user_input}"
            f"根据大纲完成完整剧本的创作或修改剧本，要求根据所给的内容输出作品的完整剧本,"
            f"并进行合理分段，并保证剧情连贯。要求内容完整充实，分段合理。"
            f"对不同的分段之间应保证剧情的连贯性。分段要求：如地点变化必须新分一段。"
            f"同一段中场景地点必须只有一个。以下为原大纲内容{get_outline}\n"
            f"{_CHARACTER_NAME_LOCK}"
        )
    
    result = generator.generate_with_retry(
        user_input=user_input,
        strict_model=strict_model,
        content_formatter=format_script_input
    )
    
    tools.save_dict_to_json(result.model_dump(), path)
    return result.model_dump()
