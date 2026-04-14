from typing import Any, Dict, List, Literal, Optional, get_args

from pydantic import BaseModel, Field

from app.services.dialogue_stand_assets import (
    NARRATION_CHARACTER_TOKEN,
    SPIRIT_CHARACTER_TOKEN,
    build_cast_character_instruction_block,
    normalize_story_cast,
    rewrite_dialogue_character_to_paths,
)
from app.services.llm_generator import LLMGenerator
from app.services import tools

CharacterExpression = Literal[
    "happy",
    "depression",
    "crying",
    "smile",
    "surprise",
    "shy",
    "tsundere",
    "anger",
    "sad",
]
expression_ls = list(get_args(CharacterExpression))

_CHARACTER_NAME_LOCK = (
    "【姓名锁定】说话者 name 必须与剧本及用户设定中的正式角色姓名逐字一致；禁止翻译、改写或替换为近义称呼。"
)

# 与流程图 site_description 同源：结构化输出中的 site 字段
_SITE_NO_PEOPLE_RULE = (
    "【地点描述·禁止人物】输出中的 site（章节地点/场景说明，对应下游 site_description）"
    "只写环境、空间、陈设、光线、氛围、天气等纯场景信息；"
    "严禁出现任何人物、角色、人称、对话者、身体部位、衣着穿戴在角色身上的描写；"
    "勿用「有人」「某人」「他/她」「主角」等暗示在场者的表述；若原文含人物活动，改写为无人的空景或静物视角。"
)


def dialogue(
    user_input: str,
    script,
    save_path: str,
    strict_model: bool = False,
    story_cast: Optional[List[Dict[str, Any]]] = None,
):
    """
    :param story_cast: 故事页已选角色 [{\"character_name\": \"...\"}]；非空时 character 按各角色目录下 .txt 文件名（stem）
        由模型选择，再替换为 /sources/pic/... 路径；旁白 character 置空；非名单角色用通用精灵图路径。
    """
    cast_entries = normalize_story_cast(story_cast)
    if cast_entries:
        cast_block = build_cast_character_instruction_block(cast_entries)

        class DialogueItemCast(BaseModel):
            name: str = Field(
                description="说话者的名字；须与剧本中该角色正式姓名逐字一致。若是旁白则写'旁白'"
            )
            dialogue_content: str = Field(description="该说话者说的内容，合理分句，单句应不超过15字")
            character: str = Field(
                description=(
                    "立绘标识字符串：须与上文「仅 name=某角色名」块中该角色下列出的某一标识完全一致；"
                    "不同角色的可选集合不同，禁止把甲角色的标识用于乙角色。"
                    f"旁白必须恰好填 {NARRATION_CHARACTER_TOKEN}；"
                    f"其他非名单说话者填 {SPIRIT_CHARACTER_TOKEN}"
                )
            )

        class DialogueCast(BaseModel):
            dialogues: List[DialogueItemCast] = Field(description="具体的对话列表")
            chapter_name: str = Field(description="这个章节的名称")
            site: str = Field(
                description="本章节唯一地点/场景的纯环境说明（无人物、无角色、无「有人」类表述），用于下游 site_description"
            )

        generator = LLMGenerator(pydantic_object=DialogueCast, result_type="对话形式台本")
        return _run_dialogue_paragraphs(
            generator,
            user_input=user_input,
            script=script,
            save_path=save_path,
            strict_model=strict_model,
            cast_entries=cast_entries,
            cast_block=cast_block,
        )

    class DialogueItem(BaseModel):
        name: str = Field(
            description="说话者的名字；须与剧本中该角色正式姓名逐字一致（含中英文与标点），不可译名或改名。若是旁白则写'旁白'"
        )
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
        site: str = Field(
            description="本章节唯一地点/场景的纯环境说明（无人物、无角色、无「有人」类表述），用于下游 site_description"
        )

    generator = LLMGenerator(pydantic_object=Dialogue, result_type="对话形式台本")
    return _run_dialogue_paragraphs(
        generator,
        user_input=user_input,
        script=script,
        save_path=save_path,
        strict_model=strict_model,
        cast_entries=None,
        cast_block=None,
    )


def _legacy_format_dialogue_input(user_input: str, content_i: str) -> str:
    expr_hint = "、".join(expression_ls)
    return (
        f"用户要求或待修改对话：{user_input}"
        f"{_CHARACTER_NAME_LOCK}"
        f"""根据剧本完成对话形式台本的创作或修改对话，要求根据所给的内容输出作品的完整对话形式台本,
                形式如：【（说话者的名字，若是旁白则写'旁白'） ： （该说话者说的内容）】
                       【。。。】
                       #合理分句，单句应不超过15字！！
                       #每条对话在结构化输出中必须包含 character 字段：表示说话者此时的表情，只能从以下候选中精确选一项（勿自造词）：{expr_hint}。
                       #旁白若需表情，可选用与叙述氛围最贴近的一项（如平静叙述可用「微笑」等）。
                       #内容不应过少，可适当扩充对话，丰富人物形象，但应保证前后的连贯性。
                       #并完成这个章节故事的地点的详细描述，如名称、地点、装饰、场景等等。（应有且仅有一个地点，若没有或原文不止一个，则自由发挥确保输出应有且仅有一个地点）
                       #{_SITE_NO_PEOPLE_RULE}
                以下为原剧本内容{content_i}\n请根据原剧本内容创作"""
    )


def _cast_format_dialogue_input(user_input: str, content_i: str, cast_block: str) -> str:
    return (
        f"用户要求或待修改对话：{user_input}\n"
        f"{_CHARACTER_NAME_LOCK}\n\n"
        f"{cast_block}\n\n"
        f"""根据剧本完成对话形式台本的创作或修改对话，要求根据所给的内容输出作品的完整对话形式台本,
                形式如：【（说话者的名字，若是旁白则写'旁白'） ： （该说话者说的内容）】
                       【。。。】
                       #合理分句，单句应不超过15字！！
                       #每条对话必须包含 character 字段，取值规则见上文【立绘标识 character 规则】。
                       #每条对话先根据 name 锁定「哪一名角色」：character 只能从该角色专属块所列标识中选，不得使用其他角色块中的标识。
                       #内容不应过少，可适当扩充对话，丰富人物形象，但应保证前后的连贯性。
                       #并完成这个章节故事的地点的详细描述，如名称、地点、装饰、场景等等。（应有且仅有一个地点，若没有或原文不止一个，则自由发挥确保输出应有且仅有一个地点）
                       #{_SITE_NO_PEOPLE_RULE}
                以下为原剧本内容{content_i}\n请根据原剧本内容创作"""
    )


def _run_dialogue_paragraphs(
    generator: LLMGenerator,
    *,
    user_input: str,
    script: dict,
    save_path: str,
    strict_model: bool,
    cast_entries: Optional[List[Dict[str, str]]],
    cast_block: Optional[str],
) -> List[dict]:
    paragraph_num = script["paragraph_num"]
    content = script["content"]
    re_result: List[dict] = []

    for i in range(paragraph_num):
        paragraph_text = content[i]

        def content_formatter(ui: str, pr: str, pt: str = paragraph_text) -> str:
            if cast_block is not None:
                return _cast_format_dialogue_input(ui, pt, cast_block)
            return _legacy_format_dialogue_input(ui, pt)

        result = generator.generate_with_retry(
            user_input=user_input,
            strict_model=strict_model,
            content_formatter=content_formatter,
            final_temperature=0.3,
        )
        dumped = result.model_dump()
        if cast_entries is not None:
            rewrite_dialogue_character_to_paths(dumped.get("dialogues") or [], cast_entries)
        re_result.append(dumped)

    tools.save_dict_to_json(re_result, save_path)
    return re_result
