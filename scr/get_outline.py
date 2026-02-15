import llm_chat
import const
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import tools


class ArticleOutline(BaseModel):
    article_outline_name: str = Field(description="文章大纲的名字")
    content: str = Field(description="文章大纲的内容")


parser = PydanticOutputParser(pydantic_object=ArticleOutline)
prompt = PromptTemplate(
    template="""你是一个作家。请根据所给的内容输出作品的大纲：

    创作要求：{text}

    请严格按照以下格式输出,不要添加任何额外说明：：
    {format_instructions}

    输出：""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = llm_chat.LlmChat(
    model_name='deepseek-reasoner',
    temperature=1.5,
    model_provider='openai',
    base_url='https://api.deepseek.com',
    api_key=const.api_key,
    pydantic_object=ArticleOutline,
    prompt_template=prompt
)

llm.new_message = ''
# 完成一篇科幻小说的大纲
strict_model = False
while True:
    user_input = input('input:')
    if user_input == 'exit':
        x = llm.new_message
        result = llm.structured_chat(str(x)+'\n'+'以上为最终确定的大纲，按我要求的格式输出大纲的标题与内容，不要大纲内容进行删改，也不要有多余内容')
        print(result)
        tools.save_dict_to_json(result.model_dump(), f'../data/outline/{const.time_now_}.json')
        break
    else:
        llm.singe_chat(f'完成或修改作品的大纲。用户要求:{user_input}.\n原大纲:{llm.new_message}（可能为空，按用户要求完成大纲即可）')
        while True:
            eva = llm.evaluate_result(llm.new_message, '大纲')
            if eva.res == "Perfect" or (eva.res == "Good" and not strict_model):
                print(llm.new_message)
                break
            else:
                llm.singe_chat(f'完成文章大纲的修改。原大纲:{llm.new_message}.\n用户原始要求（务必参考）{user_input}\n其他修改建议:{eva.reason_and_advise}')





