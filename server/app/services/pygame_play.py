"""
Galgame 式结构化剧本浏览（pygame）。

- 按 JSON 的 parent_id / children 与跨块引用 g{块}:{id} 在图上游走。
- setOrChangeFlag：进入节点时按「变量 = 值」写入状态（如 b = 3）。
- checkFlag：子节点键为 children 中的引用，值为条件字符串；与当前变量状态匹配则走该分支。
- menu：branch 时在中间显示按钮（content）；点击后先执行对应 flag（赋值），再按 checkFlag 选子节点。

入口：run_galgame_script(json_path)
"""
from __future__ import annotations

import copy
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from app.services.get_runninghub_pic import list_stand_pic_png_paths
from app.services.tools import get_project_root, read_json_file

_CROSS_BLOCK_REF = re.compile(r"^g(\d+):(.+)$")


def _sanitize_path_segment(name: str) -> str:
    s = (name or "").strip()
    for c in '<>:"/\\|?*\n\r\t':
        s = s.replace(c, "_")
    return s.strip(" .") or "unnamed"


@dataclass
class GalgameLine:
    """单条对白/旁白，字段与 dialogue_content 节点一致。"""

    id: str = ""
    name: str = ""
    content: str = ""
    background: str = ""
    character: str = ""
    music: str = ""
    sound: str = ""
    transition: str = ""
    menu: list = field(default_factory=list)
    setOrChangeFlag: str = ""
    checkFlag: Any = ""
    branch_num: int = 1
    parent_id: str = ""
    children: list = field(default_factory=list)
    block_index: int = 0

    @classmethod
    def from_dict(cls, d: dict, block_index: int = 0) -> GalgameLine:
        if not isinstance(d, dict):
            return cls(block_index=block_index)
        cf = d.get("checkFlag", "")
        if isinstance(cf, dict):
            check_norm = {str(k): v for k, v in cf.items()}
        else:
            check_norm = cf
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "") or ""),
            content=str(d.get("content", "") or ""),
            background=str(d.get("background", "") or ""),
            character=str(d.get("character", "") or ""),
            music=str(d.get("music", "") or ""),
            sound=str(d.get("sound", "") or ""),
            transition=str(d.get("transition", "") or ""),
            menu=list(d.get("menu") or []),
            setOrChangeFlag=str(d.get("setOrChangeFlag", "") or ""),
            checkFlag=check_norm,
            branch_num=int(d.get("branch_num") or 1),
            parent_id=str(d.get("parent_id", "") or "").strip(),
            children=list(d.get("children") or []),
            block_index=block_index,
        )


def _resolve_json_path(json_path: str) -> Path:
    p = Path(json_path)
    if p.is_file():
        return p.resolve()
    root = get_project_root()
    cand = root / json_path
    if cand.is_file():
        return cand.resolve()
    cand = Path(os.getcwd()) / json_path
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"找不到剧本 JSON：{json_path}")


def _resolve_background_path(project_root: Path, background: str) -> Optional[Path]:
    s = (background or "").strip()
    if not s:
        return None
    s = s.replace("\\", "/").strip()
    if s.startswith("/"):
        s = s.lstrip("/")
    pub = project_root / "public"
    full = pub / s
    if full.is_file():
        return full
    full2 = Path(s)
    if full2.is_file():
        return full2.resolve()
    return None


def _character_sprite_path(project_root: Path, speaker_name: str, character: str) -> Optional[Path]:
    """
    立绘路径：
    - character 为站点路径（/sources/... 或 sources/...）时，直接解析 project_root/public 下文件；
    - 否则：public/sources/pic/{角色名}/ 下，与 RunningHub / 去背景一致，认 {expr}.png、{expr}_1.png 等。
    """
    ch = (character or "").strip()
    if not ch:
        return None
    norm = ch.replace("\\", "/").strip()
    if norm.startswith("/sources/") or norm.startswith("sources/"):
        rel = norm.lstrip("/")
        cand = project_root / "public" / rel
        if cand.is_file():
            return cand.resolve()
        return None
    folder = _sanitize_path_segment(speaker_name)
    pic_dir = project_root / "public" / "sources" / "pic" / folder
    if not pic_dir.is_dir():
        return None

    if ch.lower().endswith(".png"):
        full = pic_dir / ch
        if full.is_file():
            return full
        expr_for_match = ch[:-4]
    else:
        full = pic_dir / f"{ch}.png"
        if full.is_file():
            return full
        expr_for_match = ch

    candidates = list_stand_pic_png_paths(str(pic_dir), expr_for_match)
    if candidates:
        return Path(candidates[0])
    return None


def _parse_child_ref(ref: str, current_block: int) -> Tuple[int, str]:
    s = str(ref).strip()
    m = _CROSS_BLOCK_REF.match(s)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return current_block, s


def _norm_key(s: str) -> str:
    return (s or "").strip()


def _coerce_value(v_raw: str) -> Any:
    v_raw = _norm_key(v_raw)
    if not v_raw:
        return ""
    try:
        if "." in v_raw:
            return float(v_raw)
        return int(v_raw)
    except ValueError:
        return v_raw


def apply_flag_str(state: Dict[str, Any], flag_s: str) -> None:
    """解析并应用「变量 = 值」到 state。"""
    flag_s = (flag_s or "").strip()
    if not flag_s or "=" not in flag_s:
        return
    left, right = flag_s.split("=", 1)
    k = _norm_key(left)
    if not k:
        return
    state[k] = _coerce_value(right)


def checkflag_matches(state: Dict[str, Any], cond_s: str) -> bool:
    """cond_s 形如 'b = 3'、' a= 2'，表示当前 state 中该变量是否等于该值。"""
    cond_s = (cond_s or "").strip()
    if not cond_s or "=" not in cond_s:
        return False
    left, right = cond_s.split("=", 1)
    k = _norm_key(left)
    expected = _coerce_value(right)
    actual = state.get(k)
    if actual == expected:
        return True
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    return _norm_key(str(actual)) == _norm_key(str(expected))


def _pick_child_ref(line: GalgameLine, state: Dict[str, Any]) -> Optional[str]:
    children = [str(c).strip() for c in (line.children or []) if str(c).strip()]
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    cf = line.checkFlag if isinstance(line.checkFlag, dict) else {}
    for cref in children:
        cond = cf.get(cref)
        if cond is None:
            cond = cf.get(str(cref))
        if cond is not None and str(cond).strip() and checkflag_matches(state, str(cond)):
            return cref
    return children[0]


def _load_graph(
    json_path: str,
) -> Tuple[Dict[Tuple[int, str], GalgameLine], int, str]:
    path = _resolve_json_path(json_path)
    data = read_json_file(str(path))
    blocks = data if isinstance(data, list) else [data]
    node_map: Dict[Tuple[int, str], GalgameLine] = {}
    for bi, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        for item in block.get("dialogue_content") or []:
            if isinstance(item, dict):
                ln = GalgameLine.from_dict(item, bi)
                nid = str(ln.id).strip()
                if nid:
                    node_map[(bi, nid)] = ln
    if not node_map:
        raise ValueError("JSON 中未找到任何 dialogue_content 节点")
    # 起点：块 0 中 parent_id 为空的第一个节点（按文件顺序）
    block0_nodes = [ln for (b, _), ln in node_map.items() if b == 0]
    block0_nodes.sort(key=lambda x: str(x.id))
    root = None
    for ln in block0_nodes:
        if not (ln.parent_id or "").strip():
            root = ln
            break
    if root is None and block0_nodes:
        root = block0_nodes[0]
    if root is None:
        raise ValueError("无法在块 0 确定起始节点")
    return node_map, root.block_index, str(root.id)


def _wrap_text_lines(font: pygame.font.Font, text: str, max_width: int) -> List[str]:
    text = text or ""
    if not text:
        return []
    out: List[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        w, _ = font.size(trial)
        if w <= max_width:
            current = trial
        else:
            if current:
                out.append(current)
            current = ch
    if current:
        out.append(current)
    return out


class GalgameViewer:
    """负责加载资源并按当前 GalgameLine 绘制；可选绘制居中 menu 按钮。"""

    def __init__(self, size: tuple[int, int]):
        self.w, self.h = size
        self.project_root = get_project_root()
        self.pub_word = self.project_root / "public" / "word"
        font_path = self.pub_word / "Aa楷体.ttf"
        bg_path = self.pub_word / "word_bg.png"
        if not font_path.is_file():
            raise FileNotFoundError(f"缺少字体文件：{font_path}")
        if not bg_path.is_file():
            raise FileNotFoundError(f"缺少字幕背景：{bg_path}")
        self._font_path = str(font_path)
        self._word_bg_orig = pygame.image.load(str(bg_path)).convert_alpha()
        self._bg_surface: Optional[pygame.Surface] = None
        self._bg_key: Optional[str] = None
        self._char_surface: Optional[pygame.Surface] = None
        self._char_key: Optional[str] = None

    def _font(self, px: int) -> pygame.font.Font:
        return pygame.font.Font(self._font_path, px)

    def _ensure_background(self, line: GalgameLine) -> None:
        key = line.background or ""
        if key == self._bg_key and self._bg_surface is not None:
            return
        self._bg_key = key
        path = _resolve_background_path(self.project_root, line.background)
        if path and path.is_file():
            img = pygame.image.load(str(path)).convert()
            iw, ih = img.get_size()
            scale = min(self.w / iw, self.h / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            scaled = pygame.transform.smoothscale(img, (nw, nh))
            surf = pygame.Surface((self.w, self.h))
            surf.fill((0, 0, 0))
            x = (self.w - nw) // 2
            y = (self.h - nh) // 2
            surf.blit(scaled, (x, y))
            self._bg_surface = surf
        else:
            self._bg_surface = pygame.Surface((self.w, self.h))
            self._bg_surface.fill((16, 18, 28))

    def _ensure_character(self, line: GalgameLine) -> None:
        key = f"{line.name}|{line.character}"
        if key == self._char_key:
            return
        self._char_key = key
        path = _character_sprite_path(self.project_root, line.name, line.character)
        if not path:
            self._char_surface = None
            return
        try:
            img = pygame.image.load(str(path)).convert_alpha()
        except pygame.error:
            self._char_surface = None
            return
        max_h = int(self.h * 0.88)
        iw, ih = img.get_size()
        if ih > max_h and ih > 0:
            scale = max_h / ih
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            img = pygame.transform.smoothscale(img, (nw, nh))
        self._char_surface = img

    def layout_menu_buttons(
        self, menu_items: List[dict], max_buttons: int = 12
    ) -> List[pygame.Rect]:
        """在画面中部纵向排列按钮区域，返回与 menu_items 一一对应的 Rect（最多 max_buttons）。"""
        rects: List[pygame.Rect] = []
        n = min(len(menu_items), max_buttons)
        if n <= 0:
            return rects
        bw = min(520, int(self.w * 0.55))
        bh = 48
        gap = 14
        total_h = n * bh + (n - 1) * gap
        start_y = (self.h - total_h) // 2
        x0 = (self.w - bw) // 2
        for i in range(n):
            rects.append(pygame.Rect(x0, start_y + i * (bh + gap), bw, bh))
        return rects

    def draw(
        self,
        screen: pygame.Surface,
        line: GalgameLine,
        *,
        waiting_menu: bool,
        menu_rects: List[pygame.Rect],
        vars_display: Dict[str, Any],
        hint_text: str,
    ) -> None:
        self._ensure_background(line)
        self._ensure_character(line)
        if self._bg_surface:
            screen.blit(self._bg_surface, (0, 0))
        if self._char_surface:
            sw, sh = self._char_surface.get_size()
            margin_x = 32
            bottom_margin = 8
            x = margin_x
            y = self.h - sh - bottom_margin
            y = max(0, y)
            screen.blit(self._char_surface, (x, y))

        if waiting_menu and menu_rects and line.menu:
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            title_font = self._font(24)
            title = title_font.render("请选择", True, (240, 248, 255))
            screen.blit(title, ((self.w - title.get_width()) // 2, menu_rects[0].y - 44))
            btn_font = self._font(22)
            for i, rect in enumerate(menu_rects):
                if i >= len(line.menu):
                    break
                item = line.menu[i]
                label = ""
                if isinstance(item, dict):
                    label = str(item.get("content", "") or "")
                else:
                    label = str(item)
                pygame.draw.rect(screen, (40, 48, 72), rect, border_radius=10)
                pygame.draw.rect(screen, (100, 180, 255), rect, 2, border_radius=10)
                text_s = btn_font.render(label[:40] or f"选项{i + 1}", True, (255, 255, 255))
                tx = rect.x + (rect.w - text_s.get_width()) // 2
                ty = rect.y + (rect.h - text_s.get_height()) // 2
                screen.blit(text_s, (tx, ty))

        # 字幕条与屏幕同宽；左侧多留空，避免压住 word_bg 装饰边框
        bar_target_w = self.w
        bar_h = int(self.h * 0.22)
        bar_h = max(120, min(bar_h, 220))
        scaled_bar = pygame.transform.smoothscale(
            self._word_bg_orig, (bar_target_w, bar_h)
        )
        bar_x = 0
        bar_y = self.h - bar_h
        screen.blit(scaled_bar, (bar_x, bar_y))

        pad_y = 16
        pad_left = max(72, int(self.w * 0.055))
        pad_right = max(40, int(self.w * 0.03))
        text_x = bar_x + pad_left
        inner_w = max(120, bar_target_w - pad_left - pad_right)

        # 浅黄底上用深灰字，保证对比与可读性
        name_color = (38, 38, 46)
        body_color = (42, 42, 50)
        vars_color = (72, 72, 80)
        hint_color = (88, 88, 98)

        name_font = self._font(26)
        body_font = self._font(22)

        ty = bar_y + pad_y
        name_text = line.name.strip() or " "
        name_s = name_font.render(name_text, True, name_color)
        screen.blit(name_s, (text_x, ty))
        ty += name_s.get_height() + 8

        content_lines = _wrap_text_lines(body_font, line.content, inner_w)
        for ln in content_lines[:5]:
            surf = body_font.render(ln, True, body_color)
            screen.blit(surf, (text_x, ty))
            ty += surf.get_height() + 4

        if vars_display:
            vf = self._font(16)
            vs = ", ".join(f"{k}={v}" for k, v in sorted(vars_display.items()))
            if len(vs) > 80:
                vs = vs[:77] + "..."
            v_surf = vf.render(vs, True, vars_color)
            screen.blit(v_surf, (text_x, ty + 4))

        hint = self._font(18).render(hint_text, True, hint_color)
        screen.blit(hint, (text_x, bar_y + bar_h - pad_y - hint.get_height()))


def run_galgame_script(json_path: str) -> None:
    node_map, start_bi, start_id = _load_graph(json_path)
    state: Dict[str, Any] = {}
    current_block = start_bi
    current_id = start_id
    history: List[Tuple[int, str, Dict[str, Any]]] = []

    def current_line() -> GalgameLine:
        return node_map[(current_block, current_id)]

    def enter_node(bi: int, nid: str) -> None:
        nonlocal current_block, current_id, state
        current_block, current_id = bi, nid
        ln = node_map[(bi, nid)]
        apply_flag_str(state, ln.setOrChangeFlag)
        history.append((bi, nid, copy.deepcopy(state)))

    def navigate_to(nb: int, nn: str) -> None:
        """离开当前节点前刷新栈顶变量快照（含菜单点击后的 flag），再进入下一点。"""
        nonlocal state
        if history:
            history[-1] = (current_block, current_id, copy.deepcopy(state))
        if (nb, nn) not in node_map:
            return
        enter_node(nb, nn)

    enter_node(start_bi, start_id)

    pygame.init()
    pygame.display.set_caption("Galgame 剧本浏览")
    size = (1280, 720)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    viewer = GalgameViewer(size)
    menu_rects: List[pygame.Rect] = []

    def rebuild_menu_rects() -> None:
        nonlocal menu_rects
        ln = current_line()
        if (
            ln.branch_num > 1
            and ln.menu
            and isinstance(ln.menu, list)
            and len(ln.children) > 1
        ):
            menu_rects = viewer.layout_menu_buttons(ln.menu)
        else:
            menu_rects = []

    rebuild_menu_rects()

    running = True
    while running:
        ln = current_line()
        waiting_menu = bool(menu_rects)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    if len(history) > 1:
                        history.pop()
                        prev_bi, prev_id, prev_vars = history[-1]
                        current_block, current_id = prev_bi, prev_id
                        state = copy.deepcopy(prev_vars)
                        rebuild_menu_rects()
                elif not waiting_menu:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT):
                        children = [str(c).strip() for c in (ln.children or []) if str(c).strip()]
                        if not children:
                            pass
                        elif len(children) == 1:
                            nb, nn = _parse_child_ref(children[0], current_block)
                            navigate_to(nb, nn)
                            rebuild_menu_rects()
                        else:
                            cref = _pick_child_ref(ln, state)
                            if cref:
                                nb, nn = _parse_child_ref(cref, current_block)
                                navigate_to(nb, nn)
                                rebuild_menu_rects()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if waiting_menu:
                    for i, rect in enumerate(menu_rects):
                        if rect.collidepoint(event.pos) and i < len(ln.menu):
                            item = ln.menu[i]
                            flag_s = ""
                            if isinstance(item, dict):
                                flag_s = str(item.get("flag", "") or "")
                            apply_flag_str(state, flag_s)
                            cref = _pick_child_ref(ln, state)
                            if cref:
                                nb, nn = _parse_child_ref(cref, current_block)
                                navigate_to(nb, nn)
                                rebuild_menu_rects()
                            break
                else:
                    children = [str(c).strip() for c in (ln.children or []) if str(c).strip()]
                    if not children:
                        pass
                    elif len(children) == 1:
                        nb, nn = _parse_child_ref(children[0], current_block)
                        navigate_to(nb, nn)
                        rebuild_menu_rects()
                    else:
                        cref = _pick_child_ref(ln, state)
                        if cref:
                            nb, nn = _parse_child_ref(cref, current_block)
                            navigate_to(nb, nn)
                            rebuild_menu_rects()

        ln = current_line()
        waiting_menu = bool(menu_rects)
        if waiting_menu:
            hint = "点击按钮选择  ·  ← 返回上一节点  ·  Esc 退出"
        else:
            children = [str(c).strip() for c in (ln.children or []) if str(c).strip()]
            if not children:
                hint = "终点  ·  ← 返回  ·  Esc 退出"
            elif len(children) > 1 and not ln.menu:
                hint = "空格/点击：按变量与 checkFlag 自动选分支  ·  ← 返回  ·  Esc"
            else:
                hint = "空格/点击 下一句  ·  ← 返回  ·  Esc 退出"

        viewer.draw(
            screen,
            ln,
            waiting_menu=waiting_menu,
            menu_rects=menu_rects,
            vars_display=state,
            hint_text=hint,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    if len(sys.argv) < 2:
        root = get_project_root()
        default = root / "public" / "sources" / "strctured_json" / "test1111.json"
        path = str(default)
        print(f"未传入路径，使用默认：{path}")
    else:
        path = sys.argv[1]
    run_galgame_script(path)


if __name__ == "__main__":
    main()
