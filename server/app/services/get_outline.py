from app.services.llm_generator import LLMGenerator
from pydantic import BaseModel, Field
from app.services import tools

# 与用户数据中「角色姓名」逐字一致，避免模型擅自翻译/改名
_CHARACTER_NAME_LOCK = (
    "【姓名锁定】若用户数据中给出了角色的正式姓名（如「角色姓名」字段或「【角色：…】」标题行），"
    "则大纲中出现的该角色称呼必须与之一字不差（含中英文、数字、标点、空格）；禁止自行翻译（例如英文译中文）、改名、缩写，"
    "或用昵称/代称替代正式姓名；旁白与角色间称呼亦须遵守。"
)


def outline(user_input: str, save_path: str, strict_model: bool = False):
    class ArticleOutline(BaseModel):
        article_outline_name: str = Field(description="文章大纲的名字")
        content: str = Field(description="文章大纲的内容")
    
    generator = LLMGenerator(
        pydantic_object=ArticleOutline,
        result_type='大纲'
    )
    
    def format_outline_input(user_input, previous_result):
        return (
            f"完成或修改作品的大纲。用户数据:{user_input}.\n"
            f"原大纲:{previous_result}（可能为空，按用户要求完成大纲即可）\n"
            f"{_CHARACTER_NAME_LOCK}"
        )
    
    result = generator.generate_with_retry(
        user_input=user_input,
        strict_model=strict_model,
        content_formatter=format_outline_input
    )
    
    print(result)
    tools.save_dict_to_json(result.model_dump(), save_path)
    return result.model_dump()
