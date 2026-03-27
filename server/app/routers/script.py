from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.get_complete_script import complete_script
from app.services import tools

router = APIRouter(prefix="/api/script", tags=["script"])


class ScriptRequest(BaseModel):
    user_input: str
    outline: str
    strict_model: bool = False


class ScriptResponse(BaseModel):
    article_script_name: str
    paragraph_num: int
    content: List[str]
    site: str


@router.post("/generate", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    try:
        save_path = tools.generate_save_path("complete_script", "script")
        
        result = complete_script(
            user_input=request.user_input,
            outline=request.outline,
            path=save_path,
            strict_model=request.strict_model
        )
        
        return ScriptResponse(
            article_script_name=result.get("article_script_name", ""),
            paragraph_num=result.get("paragraph_num", 0),
            content=result.get("content", []),
            site=result.get("site", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
