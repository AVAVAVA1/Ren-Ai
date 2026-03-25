from typing import List, Dict, Any, Optional, Tuple

from app.services import tools

_DEFAULT_EXPRESSION = "微笑"


def structured_json(
    data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    将对话生成结果转为流程图用 JSON，并写入 public/sources/strctured_json。
    :param data: 与 dialogue 输出一致，每项含 chapter_name、site、dialogues
    :param save_path: 若为空则自动生成带时间戳路径
    :return: (renai_data, 磁盘绝对路径)
    """
    renai_data: List[Dict[str, Any]] = []
    counter = 0

    for element in data:
        chapter_data = {
            "dialogue_name": element["chapter_name"],
            "site_description": element["site"],
            "dialogue_content": [],
        }
        dialogues = element.get("dialogues") or []
        total_dialogues = len(dialogues)
        for index, dialogue in enumerate(dialogues):
            parent_id = "" if index == 0 else f"{counter - 1}"
            children = [] if index == total_dialogues - 1 else [f"{counter + 1}"]

            entry = {
                "id": f"{counter}",
                "name": dialogue["name"],
                "content": dialogue["dialogue_content"],
                "background": "",
                "character": dialogue.get("character") or _DEFAULT_EXPRESSION,
                "music": "",
                "sound": "",
                "transition": "",
                "menu": [],
                "setOrChangeFlag": "",
                "checkFlag": "",
                "branch_num": 1,
                "parent_id": parent_id,
                "children": children,
            }
            chapter_data["dialogue_content"].append(entry)
            counter += 1

        renai_data.append(chapter_data)

    if not save_path:
        save_path = tools.generate_save_path("strctured_json", "renai")
    tools.save_dict_to_json(data=renai_data, file_path=save_path)
    return renai_data, save_path
