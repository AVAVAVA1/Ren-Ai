import tools
import json_to_renpy

list_src = []
text = tools.read_json_file('../data/dialogue/0dialogue2026-02-02 22_28_06.json')
site_description = text.get('site', '')
name = text['chapter_name']
text = text['dialogues']

parent_id = ''
children = []
for i in range(len(text)):
    if i != 0:
        parent_id = str(i - 1)
    if i != len(text) - 1:
        children = [str(i + 1)]
    else:
        children = []
    x = {
        "id": str(i),
        "name": text[i]['name'],
        "content": text[i]['dialogue_content'],
        "branch_num": 1,
        "parent_id": parent_id,
        "children": children
    }
    list_src.append(x)

tools.save_dict_to_json({"dialogue_name": name, "site_description": site_description, "dialogue_content": list_src},
                        '../resource/label2.json')

# 自动转换为Ren'Py脚本
json_to_renpy.convert_json_file('../resource/label2.json', '../resource/label2.rpy')
