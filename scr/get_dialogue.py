import llm_chat
import const
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import tools

# 首先定义对话项的子模型
class DialogueItem(BaseModel):
    name: str = Field(description="说话者的名字。若是旁白则写‘旁白’")
    dialogue_content: str = Field(description="该说话者说的内容，合理分句，单句应不超过15字")


# 然后定义主模型，包含对话项列表
class Dialogue(BaseModel):
    dialogues: List[DialogueItem] = Field(description="具体的对话列表")
    chapter_name: str = Field(description="这个章节的名称")
    site: str = Field(description="这个章节故事的地点的详细描述")


parser = PydanticOutputParser(pydantic_object=Dialogue)
prompt = PromptTemplate(
    template="""你是一个作家。请根据所给的剧本内容输出对话形式台本的创作：
    #合理分句，单句应不超过20字！！
    #内容不应过少，可适当扩充对话，丰富人物形象，但应保证前后的连贯性。
    #并完成这个章节故事的地点的详细描述，如名称、地点、装饰、场景等等。（应有且仅有一个地点，若没有或原文不止一个，则自由发挥确保输出应有且仅有一个地点）
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
    pydantic_object=Dialogue,
    prompt_template=prompt
)
strict_model = False
path = '2026-02-02 22_28_06.json'
get_dialogue = tools.read_json_file(f'../data/complete_script/{path}')
paragraph_num = get_dialogue['paragraph_num']
content = get_dialogue['content']
article_script_name = get_dialogue['article_script_name']
for i in range(paragraph_num):
    llm.change_temperature(1.5)
    llm.new_message = ''
    while True:
        user_input = f'''根据剧本完成对话形式台本的创作，要求根据所给的内容输出作品的完整对话形式台本,
        形式如：【（说话者的名字，若是旁白则写‘旁白’） ： （该说话者说的内容）】
               【。。。】
               #合理分句，单句应不超过20字！！
               #内容不应过少，可适当扩充对话，丰富人物形象，但应保证前后的连贯性。
               #并完成这个章节故事的地点的详细描述，如名称、地点、装饰、场景等等。（应有且仅有一个地点，若没有或原文不止一个，则自由发挥确保输出应有且仅有一个地点）
        以下为原剧本内容{content[i]}\n请根据原剧本内容创作'''
        llm.singe_chat(user_input)
        eva = llm.evaluate_result(llm.new_message, '对话形式台本')
        if eva.res == "Perfect" or (eva.res == "Good" and not strict_model):
            print(llm.new_message)
            x = llm.new_message
            llm.change_temperature(0.3)
            result = llm.structured_chat(str(x)+'\n'+'以上为最终确定的对话形式台本，按我要求的格式输出剧本内容，不要剧本内容进行额外删改，也不要有多余内容')
            tools.save_dict_to_json(result.model_dump(), f'../data/dialogue/{i}dialogue{path}')
            break
        else:
            llm.singe_chat(
                f'完成对话形式台本的修改。原对话形式台本:{llm.new_message}.\n用户原始要求（务必参考）{user_input}\n其他修改建议:{eva.reason_and_advise}')


