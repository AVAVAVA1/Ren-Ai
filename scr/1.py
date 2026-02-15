import llm_chat
import const
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class Article(BaseModel):
    article_name: str = Field(description="文章的名字")
    content: str = Field(description="文章的内容")


parser = PydanticOutputParser(pydantic_object=Article)
prompt = PromptTemplate(
    template="""你是一个科幻作家。请根据以下要求创作一篇文章：

    创作要求：{text}

    请严格按照以下格式输出,不要添加任何额外说明：：
    {format_instructions}

    输出：""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = llm_chat.LlmChat(
    model_name='deepseek-reasoner',
    temperature=0.0,
    model_provider='openai',
    base_url='https://api.deepseek.com',
    api_key=const.api_key,
    pydantic_object=Article,
    prompt_template=prompt
)
llm.change_temperature(0.8)
result = llm.singe_chat('写一篇科幻风格的文章，内容要求20000字左右，内容可以偏轻科幻')
print(result)
print(result.article_name)
print(result.content)