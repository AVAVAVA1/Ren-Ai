from typing import List, Dict, Any, Optional, Tuple

from app.services import tools

_DEFAULT_EXPRESSION = "微笑"


def structured_json(
    data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    *,
    persist: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    将对话生成结果转为流程图用 JSON；可选写入 public/sources/strctured_json。
    :param data: 与 dialogue 输出一致，每项含 chapter_name、site、dialogues（如 dialogue_*.json）
    :param save_path: persist 为 True 且为空则自动生成带时间戳路径
    :param persist: False 时仅返回 renai_data，不写盘
    :return: (renai_data, 磁盘绝对路径；persist False 时第二项为空字符串)
    """
    renai_data: List[Dict[str, Any]] = []

    for element in data:
        chapter_data = {
            "dialogue_name": element.get("chapter_name") or "",
            "site_description": element.get("site") or "",
            "dialogue_content": [],
        }
        dialogues = element.get("dialogues") or []
        total_dialogues = len(dialogues)
        # 每章独立 0..n-1，children/parent 仅指向本章内；FlowCanvas 用 `${groupIndex}_${id}`，不可用跨章全局递增 id
        for index, dialogue in enumerate(dialogues):
            parent_id = "" if index == 0 else str(index - 1)
            children = [] if index == total_dialogues - 1 else [str(index + 1)]

            entry = {
                "id": str(index),
                "name": dialogue.get("name") or "",
                "content": dialogue.get("dialogue_content") or "",
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

        renai_data.append(chapter_data)

    if persist:
        out_path = save_path or tools.generate_save_path("strctured_json", "renai")
        tools.save_dict_to_json(data=renai_data, file_path=out_path)
        return renai_data, out_path
    return renai_data, save_path or ""
