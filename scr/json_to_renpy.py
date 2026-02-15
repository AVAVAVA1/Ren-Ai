import json
import re
import os
from typing import Any

TRANSITION_MAP = {
    "fade": "fade",
    "dissolve": "dissolve",
    "pixellate": "pixellate",
    "vpunch": "vpunch",
    "hpunch": "hpunch",
    "blank": None,
}


def parse_resource_field(value: str) -> list[str]:
    if not value:
        return []
    value = str(value).strip()
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1]
        paths = [p.strip() for p in inner.split(',')]
        return [p for p in paths if p]
    return [value]


def parse_flag_expression(expr: str) -> list[str]:
    if not expr:
        return []
    expr = str(expr).strip()
    parts = [p.strip() for p in expr.split(',')]
    result = []
    for p in parts:
        if '=' in p and '==' not in p and '>=' not in p and '<=' not in p and '!=' not in p:
            p = p.replace('=', ' = ')
        result.append(p)
    return result


def extract_variable_name(expr: str) -> str:
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', expr.strip())
    if match:
        return match.group(1)
    return None


def extract_all_variables(dialogue_content: list[dict]) -> set[str]:
    variables = set()
    for node in dialogue_content:
        if node.get('menu'):
            for menu_item in node['menu']:
                flag = menu_item.get('flag', '')
                if flag:
                    for expr in parse_flag_expression(flag):
                        var_name = extract_variable_name(expr)
                        if var_name:
                            variables.add(var_name)
        set_flag = node.get('setOrChangeFlag', '')
        if set_flag:
            for expr in parse_flag_expression(set_flag):
                var_name = extract_variable_name(expr)
                if var_name:
                    variables.add(var_name)
        check_flag = node.get('checkFlag', {})
        if isinstance(check_flag, dict):
            for condition in check_flag.values():
                var_name = extract_variable_name(str(condition))
                if var_name:
                    variables.add(var_name)
    return variables


def extract_all_characters(dialogue_content: list[dict]) -> dict[str, str]:
    characters = {}
    for node in dialogue_content:
        name = node.get('name', '')
        if name and name != '旁白':
            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            var_name = re.sub(r'_+', '_', var_name).strip('_')
            if not var_name:
                var_name = f"char_{node['id']}"
            if name not in characters:
                characters[name] = var_name
    return characters


def build_node_map(dialogue_content: list[dict]) -> dict[str, dict]:
    return {node['id']: node for node in dialogue_content}


def find_root_node(dialogue_content: list[dict]) -> str:
    for node in dialogue_content:
        if not node.get('parent_id'):
            return node['id']
    return '0'


def get_transition(transition_value: str) -> str:
    if not transition_value:
        return None
    transition_value = str(transition_value).strip()
    return TRANSITION_MAP.get(transition_value, transition_value)


def generate_resource_commands(node: dict, indent: str = "    ") -> list[str]:
    lines = []
    bg = node.get('background', '')
    char = node.get('character', '')
    music = node.get('music', '')
    sound = node.get('sound', '')
    transition = get_transition(node.get('transition', ''))

    if bg:
        bg_paths = parse_resource_field(bg)
        for path in bg_paths:
            trans_str = f" with {transition}" if transition else ""
            lines.append(f"{indent}scene bg_{path.replace('/', '_').replace('\\\\', '_')}{trans_str}")

    if char:
        char_paths = parse_resource_field(char)
        for i, path in enumerate(char_paths):
            position = "at center" if i == 0 else f"at position_{i}"
            lines.append(f"{indent}show char_{path.replace('/', '_').replace('\\\\', '_')} {position}")

    if music:
        music_paths = parse_resource_field(music)
        for path in music_paths:
            lines.append(f'{indent}play music "{path}"')

    if sound:
        sound_paths = parse_resource_field(sound)
        for path in sound_paths:
            lines.append(f'{indent}play sound "{path}"')

    return lines


def generate_dialogue_line(node: dict, characters: dict[str, str], indent: str = "    ") -> list[str]:
    lines = []
    name = node.get('name', '')
    content = node.get('content', '')

    if not content:
        return lines

    if name == '旁白' or not name:
        lines.append(f'{indent}"{content}"')
    else:
        char_var = characters.get(name, 'narrator')
        lines.append(f'{indent}{char_var} "{content}"')

    return lines


def generate_set_flag(node: dict, indent: str = "    ") -> list[str]:
    lines = []
    set_flag = node.get('setOrChangeFlag', '')
    if set_flag:
        for expr in parse_flag_expression(set_flag):
            lines.append(f"{indent}$ {expr}")
    return lines


def generate_menu_block(node: dict, node_map: dict, characters: dict, indent: str = "    ") -> list[str]:
    lines = []
    menu_items = node.get('menu', [])
    children = node.get('children', [])

    if not menu_items:
        return lines

    lines.append(f"{indent}menu:")

    for i, item in enumerate(menu_items):
        content = item.get('content', '')
        flag = item.get('flag', '')

        lines.append(f'{indent}    "{content}":')

        if flag:
            for expr in parse_flag_expression(flag):
                lines.append(f"{indent}        $ {expr}")

        if children:
            target_id = children[0] if len(children) == 1 else children[min(i, len(children) - 1)]
            lines.append(f"{indent}        jump label_{target_id}")

    return lines


def generate_check_flag_block(node: dict, node_map: dict, indent: str = "    ") -> list[str]:
    lines = []
    check_flag = node.get('checkFlag', {})

    if not isinstance(check_flag, dict) or not check_flag:
        return lines

    conditions = []
    for node_id, condition in check_flag.items():
        cond_str = str(condition).strip()
        if '=' in cond_str and '==' not in cond_str and '>=' not in cond_str and '<=' not in cond_str and '!=' not in cond_str:
            cond_str = cond_str.replace('=', '==')
        conditions.append((cond_str, node_id))

    for i, (cond, node_id) in enumerate(conditions):
        if i == 0:
            lines.append(f"{indent}if {cond}:")
        else:
            lines.append(f"{indent}elif {cond}:")
        lines.append(f"{indent}    jump label_{node_id}")

    return lines


class RenPyGenerator:
    def __init__(self, dialogue_data: dict):
        self.dialogue_data = dialogue_data
        self.dialogue_content = dialogue_data.get('dialogue_content', [])
        self.dialogue_name = dialogue_data.get('dialogue_name', 'dialogue')
        self.node_map = build_node_map(self.dialogue_content)
        self.variables = extract_all_variables(self.dialogue_content)
        self.characters = extract_all_characters(self.dialogue_content)
        self.generated_labels = set()
        self.root_id = find_root_node(self.dialogue_content)

    def generate_header(self) -> list[str]:
        lines = []
        lines.append(f"# Generated from: {self.dialogue_name}")
        lines.append("")

        if self.variables:
            lines.append("# Variables")
            for var in sorted(self.variables):
                lines.append(f"default {var} = None")
            lines.append("")

        if self.characters:
            lines.append("# Characters")
            lines.append("define narrator = Character(None)")
            for name, var in self.characters.items():
                lines.append(f'define {var} = Character("{name}")')
            lines.append("")

        return lines

    def is_branching_node(self, node: dict) -> bool:
        return bool(node.get('menu')) or bool(node.get('checkFlag'))

    def is_merge_point(self, node_id: str) -> bool:
        parents = [n for n in self.dialogue_content if node_id in n.get('children', [])]
        return len(parents) > 1

    def generate_node_content(self, node_id: str, indent: str = "    ") -> list[str]:
        lines = []
        node = self.node_map.get(node_id)
        if not node:
            return lines

        resource_lines = generate_resource_commands(node, indent)
        lines.extend(resource_lines)

        dialogue_lines = generate_dialogue_line(node, self.characters, indent)
        lines.extend(dialogue_lines)

        set_flag_lines = generate_set_flag(node, indent)
        lines.extend(set_flag_lines)

        if node.get('menu'):
            menu_lines = generate_menu_block(node, self.node_map, self.characters, indent)
            lines.extend(menu_lines)
            return lines

        if node.get('checkFlag'):
            check_lines = generate_check_flag_block(node, self.node_map, indent)
            lines.extend(check_lines)
            return lines

        return lines

    def generate_linear_chain(self, start_id: str, indent: str = "    ") -> list[str]:
        lines = []
        current_id = start_id

        while current_id:
            if current_id in self.generated_labels:
                break

            node = self.node_map.get(current_id)
            if not node:
                break

            if self.is_branching_node(node) or self.is_merge_point(current_id):
                lines.append(f"{indent}jump label_{current_id}")
                break

            self.generated_labels.add(current_id)

            content_lines = self.generate_node_content(current_id, indent)
            lines.extend(content_lines)

            children = node.get('children', [])
            if len(children) == 0:
                lines.append(f"{indent}return")
                break
            elif len(children) == 1:
                current_id = children[0]
            else:
                break

        return lines

    def generate_branch_label(self, node_id: str) -> list[str]:
        lines = []

        if node_id in self.generated_labels:
            return lines

        self.generated_labels.add(node_id)
        node = self.node_map.get(node_id)
        if not node:
            return lines

        lines.append(f"label label_{node_id}:")

        content_lines = self.generate_node_content(node_id, "    ")
        lines.extend(content_lines)

        if self.is_branching_node(node):
            pass
        else:
            children = node.get('children', [])
            if len(children) == 1:
                chain_lines = self.generate_linear_chain(children[0], "    ")
                lines.extend(chain_lines)
            elif len(children) == 0:
                lines.append("    return")

        return lines

    def collect_branch_nodes(self) -> list[str]:
        branch_nodes = set()

        for node_id, node in self.node_map.items():
            if self.is_branching_node(node):
                branch_nodes.add(node_id)
            elif self.is_merge_point(node_id):
                branch_nodes.add(node_id)

            if node.get('menu'):
                for item in node['menu']:
                    pass

            check_flag = node.get('checkFlag', {})
            if isinstance(check_flag, dict):
                for target_id in check_flag.keys():
                    branch_nodes.add(target_id)

        return list(branch_nodes)

    def generate(self) -> str:
        all_lines = []

        header = self.generate_header()
        all_lines.extend(header)

        all_lines.append("label start:")

        main_lines = self.generate_linear_chain(self.root_id, "    ")
        all_lines.extend(main_lines)

        all_lines.append("")

        branch_nodes = self.collect_branch_nodes()
        for node_id in branch_nodes:
            label_lines = self.generate_branch_label(node_id)
            if label_lines:
                all_lines.extend(label_lines)
                all_lines.append("")

        return "\n".join(all_lines)


def convert_json_to_renpy(json_path: str, output_path: str = None) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)

    generator = RenPyGenerator(dialogue_data)
    renpy_script = generator.generate()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(renpy_script)

    return renpy_script


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <json_file> [output_file]")
        print("Example: python main.py dialogue.json output.rpy")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)

    result = convert_json_to_renpy(json_file, output_file)

    if output_file:
        print(f"RenPy script saved to: {output_file}")
    else:
        print(result)
