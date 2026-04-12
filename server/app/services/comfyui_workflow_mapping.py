"""
ComfyUI 工作流 JSON 与「注入槽位」的对应关系。

推荐：与 foo.json 同目录下放 foo.mapping.json（结构化、无歧义）。
兼容：foo.txt 使用与历史脚本相同的伪代码行（见 public/comfyui/workflow1.txt）。
无映射文件时使用内置默认节点（与 workflow1.json 一致）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services import tools

# 与早期硬编码一致（workflow1.json）
_DEFAULT_POSITIVE: List[Tuple[str, str]] = [
    ("127", "wildcard_text"),
    ("127", "populated_text"),
]
_DEFAULT_NEGATIVE: List[Tuple[str, str]] = [("45", "string")]
_DEFAULT_CHECKPOINT: Tuple[str, str] = ("79", "ckpt_name")

@dataclass
class SizeMappingSpec:
    """在 mapping.json 的 size 字段中描述如何把 width×height 写入工作流。"""

    type: str  # empty_latent | mx_slider2d
    node: str
    width_key: str = "width"
    height_key: str = "height"


_TXT_LINE_RE = re.compile(
    r"""^\s*workflow\s*\[\s*['"](?P<node>[^'"]+)['"]\s*\]\s*\[\s*['"]inputs['"]\s*\]\s*\[\s*['"](?P<input>[^'"]+)['"]\s*\]\s*=\s*(?P<slot>positive_prompt|negative_prompt|checkpoint)\s*$""",
    re.IGNORECASE,
)


@dataclass
class ComfyWorkflowMapping:
    """描述 positive / negative / checkpoint 应写入的节点与 inputs 键。"""

    positive: List[Tuple[str, str]] = field(default_factory=list)
    negative: List[Tuple[str, str]] = field(default_factory=list)
    checkpoint: Optional[Tuple[str, str]] = None
    label: str = ""
    size: Optional[SizeMappingSpec] = None

    @classmethod
    def default_builtin(cls) -> ComfyWorkflowMapping:
        return cls(
            positive=list(_DEFAULT_POSITIVE),
            negative=list(_DEFAULT_NEGATIVE),
            checkpoint=_DEFAULT_CHECKPOINT,
            label="内置默认（79/127/45）",
            size=None,
        )


def comfyui_workflows_dir() -> Path:
    return tools.get_project_root() / "public" / "comfyui"


def safe_workflow_json_path(filename: str) -> Path:
    """防止路径穿越；仅允许 public/comfyui 下的 .json 工作流文件。"""
    name = (filename or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("无效的工作流文件名")
    if not name.lower().endswith(".json"):
        raise ValueError("工作流须为 .json 文件")
    base = Path(name).name
    if base != name:
        raise ValueError("无效的工作流文件名")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not all(c in allowed for c in base):
        raise ValueError("工作流文件名仅允许字母、数字、._-")
    path = comfyui_workflows_dir() / base
    try:
        path.resolve().relative_to(comfyui_workflows_dir().resolve())
    except ValueError as e:
        raise ValueError("工作流路径非法") from e
    if not path.is_file():
        raise FileNotFoundError(f"未找到工作流：{path}")
    return path


def _parse_size_spec(raw: Any) -> Optional[SizeMappingSpec]:
    if not isinstance(raw, dict):
        return None
    t = (raw.get("type") or "").strip().lower()
    node = str(raw.get("node", "")).strip()
    if not node or not t:
        return None
    if t == "mx_slider2d":
        return SizeMappingSpec(type=t, node=node)
    if t == "empty_latent":
        wk = str(raw.get("width_key", "width")).strip() or "width"
        hk = str(raw.get("height_key", "height")).strip() or "height"
        return SizeMappingSpec(type=t, node=node, width_key=wk, height_key=hk)
    return None


def _load_mapping_json(path: Path) -> ComfyWorkflowMapping:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def slots(key: str) -> List[Tuple[str, str]]:
        v = raw.get(key)
        if v is None:
            return []
        if isinstance(v, dict):
            v = [v]
        out: List[Tuple[str, str]] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            n = str(item.get("node", "")).strip()
            inp = str(item.get("input", item.get("input_key", ""))).strip()
            if n and inp:
                out.append((n, inp))
        return out

    ck = raw.get("checkpoint")
    checkpoint: Optional[Tuple[str, str]] = None
    if isinstance(ck, dict):
        n = str(ck.get("node", "")).strip()
        inp = str(ck.get("input", "")).strip()
        if n and inp:
            checkpoint = (n, inp)

    pos = slots("positive")
    neg = slots("negative")
    if not pos or not neg or not checkpoint:
        raise ValueError(f"映射文件不完整：{path}")
    size_spec = _parse_size_spec(raw.get("size"))
    return ComfyWorkflowMapping(
        positive=pos,
        negative=neg,
        checkpoint=checkpoint,
        label=str(raw.get("label", "") or "").strip(),
        size=size_spec,
    )


def _load_mapping_txt(path: Path) -> ComfyWorkflowMapping:
    positive: List[Tuple[str, str]] = []
    negative: List[Tuple[str, str]] = []
    checkpoint: Optional[Tuple[str, str]] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _TXT_LINE_RE.match(line)
            if not m:
                raise ValueError(f"无法解析映射行：{line!r}（文件 {path}）")
            node, inp, slot = m.group("node"), m.group("input"), m.group("slot").lower()
            if slot == "positive_prompt":
                positive.append((node, inp))
            elif slot == "negative_prompt":
                negative.append((node, inp))
            else:
                checkpoint = (node, inp)
    if not positive or not negative or not checkpoint:
        raise ValueError(f".txt 映射缺少 positive / negative 或 checkpoint：{path}")
    return ComfyWorkflowMapping(
        positive=positive,
        negative=negative,
        checkpoint=checkpoint,
        label=path.stem,
        size=None,
    )


def load_workflow_mapping_for_json(workflow_json_path: Path) -> ComfyWorkflowMapping:
    """
    解析顺序：同 stem 的 .mapping.json > 同 stem 的 .txt > 内置默认。
    workflow_json_path 须为 …/foo.json。.mapping.json 损坏时回退 .txt / 内置。
    """
    stem = workflow_json_path.stem
    mapping_json = workflow_json_path.parent / f"{stem}.mapping.json"
    mapping_txt = workflow_json_path.parent / f"{stem}.txt"
    if mapping_json.is_file():
        try:
            return _load_mapping_json(mapping_json)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
            pass
    if mapping_txt.is_file():
        try:
            return _load_mapping_txt(mapping_txt)
        except (OSError, ValueError):
            pass
    return ComfyWorkflowMapping.default_builtin()


def list_comfyui_workflows() -> List[Dict[str, Any]]:
    """扫描 public/comfyui/*.json，排除 *.mapping.json。"""
    d = comfyui_workflows_dir()
    if not d.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        if p.name.endswith(".mapping.json"):
            continue
        m = load_workflow_mapping_for_json(p)
        mj = p.parent / f"{p.stem}.mapping.json"
        mt = p.parent / f"{p.stem}.txt"
        src = "builtin"
        mapping_ok = True
        mapping_error: Optional[str] = None
        if mj.is_file():
            src = "mapping.json"
            try:
                _load_mapping_json(mj)
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                mapping_ok = False
                mapping_error = str(e)
        elif mt.is_file():
            src = "txt"
            try:
                _load_mapping_txt(mt)
            except (OSError, ValueError) as e:
                mapping_ok = False
                mapping_error = str(e)
        label = m.label or p.stem
        rows.append(
            {
                "file": p.name,
                "label": label,
                "mapping_ok": mapping_ok,
                "mapping_source": src,
                "mapping_error": mapping_error,
            }
        )
    return rows


def apply_mapping_to_workflow(
    workflow: Dict[str, Any],
    mapping: ComfyWorkflowMapping,
    *,
    positive_prompt: str,
    negative_prompt: str,
    checkpoint: str,
) -> None:
    def _set(node_id: str, input_key: str, value: str) -> None:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise KeyError(f"工作流中不存在节点 {node_id!r}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise KeyError(f"节点 {node_id!r} 无 inputs 字典")
        inputs[input_key] = value

    for nid, ik in mapping.positive:
        _set(nid, ik, positive_prompt)
    for nid, ik in mapping.negative:
        _set(nid, ik, negative_prompt)
    if mapping.checkpoint:
        nid, ik = mapping.checkpoint
        _set(nid, ik, checkpoint)


def apply_workflow_dimensions(
    workflow: Dict[str, Any],
    size_spec: Optional[SizeMappingSpec],
    width: int,
    height: int,
) -> None:
    """按 mapping 中的 size 将 width/height 写入工作流（整数）。"""
    if not size_spec or width < 64 or height < 64:
        return

    def _set_int(node_id: str, input_key: str, value: int) -> None:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise KeyError(f"工作流中不存在节点 {node_id!r}")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise KeyError(f"节点 {node_id!r} 无 inputs 字典")
        inputs[input_key] = int(value)

    nid = size_spec.node
    if size_spec.type == "empty_latent":
        _set_int(nid, size_spec.width_key, width)
        _set_int(nid, size_spec.height_key, height)
    elif size_spec.type == "mx_slider2d":
        # mxSlider2D：Xi/Xf 为横向，Yi/Yf 为纵向（与 workflow1 中节点 19 一致）
        _set_int(nid, "Xi", width)
        _set_int(nid, "Xf", width)
        _set_int(nid, "Yi", height)
        _set_int(nid, "Yf", height)
    else:
        return


def read_checkpoint_default_from_workflow(
    workflow: Dict[str, Any],
    mapping: ComfyWorkflowMapping,
) -> str:
    if not mapping.checkpoint:
        return ""
    nid, ik = mapping.checkpoint
    try:
        node = workflow.get(nid) or {}
        inputs = node.get("inputs") or {}
        v = inputs.get(ik)
        return str(v).strip() if v is not None else ""
    except (TypeError, AttributeError):
        return ""
