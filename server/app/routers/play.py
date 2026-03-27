import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import tools

router = APIRouter(prefix="/api/play", tags=["play"])

RUNTIME_JSON_NAME = "_flow_play_runtime.json"

# 若 uvicorn 用系统 3.14，而 pygame 装在项目 .venv 里，请设为该 venv 的 python.exe，例如：
# C:\frontend\web_renai\.venv\Scripts\python.exe
ENV_PYGAME_PYTHON = "RENAI_PYGAME_PYTHON"


class RunFlowPygameBody(BaseModel):
    """与流程图「导出 JSON」相同结构，由 pygame_play 按图遍历。"""

    structured_json: Union[List[Any], dict] = Field(...)


def _resolve_pygame_python() -> Path:
    raw = (os.environ.get(ENV_PYGAME_PYTHON) or "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"环境变量 {ENV_PYGAME_PYTHON} 指向的路径不存在或不是文件：{raw}",
            )
        return p.resolve()
    return Path(sys.executable).resolve()


def _ensure_pygame_available(py: Path) -> None:
    server_dir = tools.get_project_root() / "server"
    try:
        r = subprocess.run(
            [str(py), "-c", "import pygame"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(server_dir.resolve()),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=503,
            detail=f"在 {py} 中检测 pygame 超时。",
        ) from None
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:800]
        raise HTTPException(
            status_code=503,
            detail=(
                f"解释器无法 import pygame：{py}。"
                f' 请执行： "{py}" -m pip install pygame'
                f"。或设置 {ENV_PYGAME_PYTHON} 指向已安装 pygame 的 Python（例如项目 .venv\\Scripts\\python.exe）。"
                + (f" 输出：{err}" if err else "")
            ),
        )


@router.post("/run-flow-pygame")
def run_flow_pygame(body: RunFlowPygameBody):
    """
    将当前结构化 JSON 写入 public/sources/strctured_json/_flow_play_runtime.json，
    并以子进程启动 pygame_play。

    默认使用与 uvicorn 相同的解释器；若 pygame 只装在别的环境，设置环境变量
    RENAI_PYGAME_PYTHON 为该环境的 python.exe 完整路径。
    """
    py_exe = _resolve_pygame_python()
    _ensure_pygame_available(py_exe)
    root = tools.get_project_root()
    out_path = root / "public" / "sources" / "strctured_json" / RUNTIME_JSON_NAME
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(body.structured_json, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"写入临时 JSON 失败: {e}") from e

    server_dir = root / "server"
    if not server_dir.is_dir():
        raise HTTPException(status_code=500, detail=f"未找到 server 目录: {server_dir}")

    try:
        subprocess.Popen(
            [
                str(py_exe),
                "-m",
                "app.services.pygame_play",
                str(out_path.resolve()),
            ],
            cwd=str(server_dir.resolve()),
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动 Pygame 子进程失败: {e}") from e

    used_env = bool((os.environ.get(ENV_PYGAME_PYTHON) or "").strip())
    return {
        "success": True,
        "message": (
            "已启动 Galgame 预览窗口。"
            f" 使用的 Python：{py_exe}"
            + (f"（环境变量 {ENV_PYGAME_PYTHON}）" if used_env else "（与当前 uvicorn 相同）")
        ),
        "json_path": str(out_path).replace("\\", "/"),
        "python_executable": str(py_exe),
    }
