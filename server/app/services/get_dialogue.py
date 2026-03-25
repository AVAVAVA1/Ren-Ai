from app.services.llm_generator import LLMGenerator
from pydantic import BaseModel, Field
from typing import List, Literal, get_args
from app.services import tools

CharacterExpression = Literal['咧嘴笑', '沮丧', '流泪', '微笑', '痛苦', '害羞', '傲娇']
expression_ls = list(get_args(CharacterExpression))


def dialogue(user_input: str, script, save_path: str, strict_model: bool = False):
    """
    :param user_input:用户输入
    :param script: 原剧本
    :param save_path: 文件保存路径
    :param strict_model: 是否严格模式
    :return: 包含所有对话的json数组
    """
    class DialogueItem(BaseModel):
        name: str = Field(description="说话者的名字。若是旁白则写'旁白'")
        dialogue_content: str = Field(description="该说话者说的内容，合理分句，单句应不超过15字")
        character: CharacterExpression = Field(
            description=(
                "说话者此时的表情描述；必须从以下候选中择一，且与台词语气、情境一致："
                + "、".join(expression_ls)
            )
        )
    class Dialogue(BaseModel):
        dialogues: List[DialogueItem] = Field(description="具体的对话列表")
        chapter_name: str = Field(description="这个章节的名称")
        site: str = Field(description="这个章节故事的地点的详细描述")
    
    generator = LLMGenerator(
        pydantic_object=Dialogue,
        result_type='对话形式台本'
    )
    
    get_dialogue = script
    paragraph_num = get_dialogue['paragraph_num']
    content = get_dialogue['content']
    article_script_name = get_dialogue['article_script_name']
    re_result = []
    
    for i in range(paragraph_num):
        def format_dialogue_input(user_input, previous_result):
            expr_hint = "、".join(expression_ls)
            return (
                f'用户要求或待修改对话：{user_input}'
                f'''根据剧本完成对话形式台本的创作或修改对话，要求根据所给的内容输出作品的完整对话形式台本,
                形式如：【（说话者的名字，若是旁白则写'旁白'） ： （该说话者说的内容）】
                       【。。。】
                       #合理分句，单句应不超过15字！！
                       #每条对话在结构化输出中必须包含 character 字段：表示说话者此时的表情，只能从以下候选中精确选一项（勿自造词）：{expr_hint}。
                       #旁白若需表情，可选用与叙述氛围最贴近的一项（如平静叙述可用「微笑」等）。
                       #内容不应过少，可适当扩充对话，丰富人物形象，但应保证前后的连贯性。
                       #并完成这个章节故事的地点的详细描述，如名称、地点、装饰、场景等等。（应有且仅有一个地点，若没有或原文不止一个，则自由发挥确保输出应有且仅有一个地点）
                以下为原剧本内容{content[i]}\n请根据原剧本内容创作'''
            )
        
        result = generator.generate_with_retry(
            user_input=user_input,
            strict_model=strict_model,
            content_formatter=format_dialogue_input,
            final_temperature=0.3
        )
        re_result.append(result.model_dump())
    
    tools.save_dict_to_json(re_result, save_path)
    return re_result
