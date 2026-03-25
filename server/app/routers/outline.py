from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.get_outline import outline

router = APIRouter(prefix="/api/outline", tags=["outline"])


class OutlineRequest(BaseModel):
    user_input: str
    strict_model: bool = False


class OutlineResponse(BaseModel):
    article_outline_name: str
    content: str


@router.post("/generate", response_model=OutlineResponse)
async def generate_outline(request: OutlineRequest):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        save_dir = os.path.join(base_dir, "public", "sources", "outline")
        os.makedirs(save_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"outline_{timestamp}.json")
        
        result = outline(
            user_input=request.user_input,
            save_path=save_path,
            strict_model=request.strict_model
        )
        
        return OutlineResponse(
            article_outline_name=result.get("article_outline_name", ""),
            content=result.get("content", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
