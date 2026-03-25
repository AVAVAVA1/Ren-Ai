from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import sys
import json
import os
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.get_outline import outline
from app.services.get_complete_script import complete_script
from app.services.get_dialogue import dialogue
from app.services.get_strctured_json import structured_json
from app.services import tools

router = APIRouter(prefix="/api/story", tags=["story"])


class StoryRequest(BaseModel):
    user_input: str
    strict_model: bool = False


class ExportStructuredRequest(BaseModel):
    """与一键生成对话阶段 dialogue() 返回的列表项结构一致（chapter_name、site、dialogues）。"""

    dialogue_results: List[Dict[str, Any]]


def format_dialogue_output(result) -> str:
    formattedContent = ''
    for item in result:
        formattedContent += f"【{item.get('chapter_name', '')}】\n地点：{item.get('site', '')}\n\n"
        dialogues = item.get("dialogues", [])
        for dialogue in dialogues:
            formattedContent += f"{dialogue.get('name', '')}：{dialogue.get('dialogue_content', '')}\n"
        formattedContent += '\n---\n\n'
    return formattedContent


@router.post("/generate")
async def generate_story(request: StoryRequest):
    async def event_stream():
        try:
            yield f"data: {json.dumps({'stage': 'start', 'message': '开始生成故事...', 'progress': 0})}\n\n"
            
            outline_save_path = tools.generate_save_path("outline", "outline")
            
            yield f"data: {json.dumps({'stage': 'outline', 'message': '正在生成大纲...', 'progress': 10})}\n\n"
            
            outline_result = outline(
                user_input=request.user_input,
                save_path=outline_save_path,
                strict_model=request.strict_model
            )
            
            outline_content = outline_result.get("content", "")
            outline_name = outline_result.get("article_outline_name", "")
            
            yield f"data: {json.dumps({'stage': 'outline_complete', 'message': '大纲生成完成', 'progress': 30, 'data': {'name': outline_name, 'content': outline_content}})}\n\n"
            
            script_save_path = tools.generate_save_path("complete_script", "script")
            
            yield f"data: {json.dumps({'stage': 'script', 'message': '正在生成剧本...', 'progress': 40})}\n\n"
            
            script_result = complete_script(
                user_input="根据大纲生成完整剧本",
                outline=outline_content,
                path=script_save_path,
                strict_model=request.strict_model
            )
            
            script_content_list = script_result.get("content", [])
            if not script_content_list:
                raise ValueError("剧本生成结果为空，无法继续生成对话")
            script_content = '\n\n'.join(script_content_list)
            script_name = script_result.get("article_script_name", "")
            script_site = script_result.get("site", "")
            # 直接使用结构化段落列表，避免 join 后再用正则切分与原始段落数不一致
            script_dict = {
                "article_script_name": script_name or "生成的剧本",
                "paragraph_num": len(script_content_list),
                "content": script_content_list,
            }
            
            yield f"data: {json.dumps({'stage': 'script_complete', 'message': '剧本生成完成', 'progress': 60, 'data': {'name': script_name, 'site': script_site, 'content': script_content}})}\n\n"
            
            dialogue_save_path = tools.generate_save_path("dialogue", "dialogue")
            
            yield f"data: {json.dumps({'stage': 'dialogue', 'message': '正在生成对话剧本...', 'progress': 70})}\n\n"
            
            dialogue_result = dialogue(
                user_input="根据剧本生成对话剧本",
                script=script_dict,
                save_path=dialogue_save_path,
                strict_model=request.strict_model
            )
            
            dialogue_output = format_dialogue_output(dialogue_result)

            dialogue_complete_payload = {
                "stage": "dialogue_complete",
                "message": "对话剧本生成完成",
                "progress": 90,
                "data": {"content": dialogue_output},
                "dialogue_results": dialogue_result,
            }
            yield f"data: {json.dumps(dialogue_complete_payload, ensure_ascii=False)}\n\n"

            complete_payload = {
                "stage": "complete",
                "message": "故事生成完成！",
                "progress": 100,
                "final_result": dialogue_output,
                "dialogue_results": dialogue_result,
            }
            yield f"data: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'message': f'生成失败: {str(e)}', 'progress': 0})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/export-structured")
async def export_structured(request: ExportStructuredRequest):
    """
    将故事生成得到的对话列表转为流程图 JSON，保存到 public/sources/strctured_json/，
    并返回前端可 fetch 的相对路径（Vite 静态资源）。
    """
    if not request.dialogue_results:
        raise HTTPException(status_code=400, detail="dialogue_results 不能为空")

    _, save_path = structured_json(request.dialogue_results)
    file_name = Path(save_path).name
    public_url = f"/sources/strctured_json/{file_name}"
    return {"public_url": public_url, "file_name": file_name, "save_path": save_path}
