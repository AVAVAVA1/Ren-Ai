from app.services.llm_generator import LLMGenerator
from pydantic import BaseModel, Field
from app.services import tools


def outline(user_input: str, save_path: str, strict_model: bool = False):
    class ArticleOutline(BaseModel):
        article_outline_name: str = Field(description="文章大纲的名字")
        content: str = Field(description="文章大纲的内容")
    
    generator = LLMGenerator(
        pydantic_object=ArticleOutline,
        result_type='大纲'
    )
    
    def format_outline_input(user_input, previous_result):
        return f'完成或修改作品的大纲。用户数据:{user_input}.\n原大纲:{previous_result}（可能为空，按用户要求完成大纲即可）'
    
    result = generator.generate_with_retry(
        user_input=user_input,
        strict_model=strict_model,
        content_formatter=format_outline_input
    )
    
    print(result)
    tools.save_dict_to_json(result.model_dump(), save_path)
    return result.model_dump()
