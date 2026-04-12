"""本地 ComfyUI HTTP API：提交 workflow、轮询 history、拉取输出图片。"""

from __future__ import annotations

import io
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from app.services import const


def _outputs_contain_image_files(outputs: Any) -> bool:
    """
    ComfyUI 在部分节点会先出现 outputs 但 images 仍为空（或 success 但无图，见 GitHub #5063）。
    仅在至少有一条带 filename 的图片记录时才认为可拉取，避免连续多任务时误判「已完成」。
    """
    if not isinstance(outputs, dict):
        return False
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        imgs = node_out.get("images")
        if not isinstance(imgs, list):
            continue
        for info in imgs:
            if isinstance(info, dict) and info.get("filename"):
                return True
    return False


class ComfyUiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")

    def queue_prompt(self, workflow: Dict[str, Any], client_id: Optional[str] = None) -> str:
        cid = client_id or str(uuid.uuid4())
        resp = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": cid},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            detail = err if isinstance(err, str) else err.get("message", err)
            raise RuntimeError(f"ComfyUI /prompt 拒绝执行: {detail}")
        pid = data.get("prompt_id")
        if not pid:
            raise RuntimeError(f"ComfyUI /prompt 未返回 prompt_id: {data}")
        return str(pid)

    def _history_record(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=60)
            if r.status_code != 200:
                return None
            j = r.json()
            if not isinstance(j, dict):
                return None
            if prompt_id in j:
                return j[prompt_id]
            if "outputs" in j or j.get("status"):
                return j
        except requests.RequestException:
            pass
        try:
            r = requests.get(f"{self.base_url}/history", params={"max_items": 200}, timeout=60)
            if r.status_code != 200:
                return None
            all_h = r.json()
            if isinstance(all_h, dict) and prompt_id in all_h:
                return all_h[prompt_id]
        except requests.RequestException:
            pass
        return None

    def wait_for_outputs(self, prompt_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = float(const.comfyui_poll_timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            rec = self._history_record(prompt_id)
            if rec:
                outs = rec.get("outputs")
                status = rec.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = rec.get("messages") or []
                    raise RuntimeError(f"ComfyUI 执行失败: {msgs}")
                # 必须已有可下载的图片元数据，不能仅凭非空 outputs 提前返回（见 ComfyUI #5063）
                if outs and _outputs_contain_image_files(outs):
                    return outs
            time.sleep(0.8)
        raise TimeoutError(
            f"ComfyUI 等待输出超时（{timeout:.0f}s），prompt_id={prompt_id}。"
            f"若单次出图超过该时间，请在 server/.env 增大 COMFYUI_POLL_TIMEOUT（当前默认 1800，单位秒）。"
        )

    def fetch_image(self, img_info: Dict[str, Any]) -> Image.Image:
        filename = img_info.get("filename")
        if not filename:
            raise ValueError(f"无效的 image 信息: {img_info}")
        params = {
            "filename": filename,
            "subfolder": img_info.get("subfolder") or "",
            "type": img_info.get("type") or "output",
        }
        r = requests.get(f"{self.base_url}/view", params=params, timeout=120)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))

    def process_workflow(self, workflow: Dict[str, Any]) -> List[Image.Image]:
        pid = self.queue_prompt(workflow)
        outputs = self.wait_for_outputs(pid)
        images: List[Image.Image] = []
        for _node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for info in node_out.get("images") or []:
                if isinstance(info, dict):
                    images.append(self.fetch_image(info))
        if not images:
            raise RuntimeError(
                "ComfyUI 已完成但未解析到图片输出；请确认工作流含 Save Image / Image Saver 等会写入输出的节点。"
            )
        return images
