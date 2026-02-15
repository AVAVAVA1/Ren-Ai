import llm_chat
import const
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import tools


class ArticleCompleteScript(BaseModel):
    article_script_name: str = Field(description="剧本的名字")
    paragraph_num: int = Field(description="剧本的分段的数目")
    content: List[str] = Field(description="剧本的完整内容，按段落分")
    site: str = Field(description="该分段的地点名")


parser = PydanticOutputParser(pydantic_object=ArticleCompleteScript)
prompt = PromptTemplate(
    template="""你是一个作家。请根据所给的内容输出作品的完整剧本,并进行合理分段，并保证剧情连贯。要求内容完整充实，分段合理。：
    分段要求：如地点变化必须新分一段。同一段中场景地点必须只有一个
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
    pydantic_object=ArticleCompleteScript,
    prompt_template=prompt
)
strict_model = False
path = '2026-02-02 22_28_06.json'
get_outline = str(tools.read_json_file(f'../data/outline/{path}'))
llm.new_message = ''
while True:
    user_input = f'根据大纲完成完整剧本的创作，要求根据所给的内容输出作品的完整剧本,并进行合理分段，并保证剧情连贯。要求内容完整充实，分段合理。对不同的分段之间应保证剧情的连贯性。分段要求：如地点变化必须新分一段。同一段中场景地点必须只有一个。以下为原大纲内容{get_outline}'
    llm.singe_chat(user_input)
    eva = llm.evaluate_result(llm.new_message, '剧本')
    if eva.res == "Perfect" or (eva.res == "Good" and not strict_model):
        print(llm.new_message)
        x = llm.new_message
        llm.change_temperature(0.0)
        result = llm.structured_chat(str(x)+'\n'+'以上为最终确定的剧本，按我要求的格式输出剧本内容，不要剧本内容进行额外删改，也不要有多余内容')
        tools.save_dict_to_json(result.model_dump(), f'../data/complete_script/{path}')
        break
    else:
        llm.singe_chat(
            f'完成剧本的修改。原剧本:{llm.new_message}.\n用户原始要求（务必参考）{user_input}\n其他修改建议:{eva.reason_and_advise}')
