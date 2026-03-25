from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.get_dialogue import dialogue
from app.services import tools

router = APIRouter(prefix="/api/dialogue", tags=["dialogue"])


class DialogueItem(BaseModel):
    dialogues: List[dict]
    chapter_name: str
    site: str


class DialogueRequest(BaseModel):
    user_input: str
    script_content: str
    strict_model: bool = False


class DialogueResponse(BaseModel):
    results: List[DialogueItem]


def parse_script_to_dict(script_text: str) -> dict:
    paragraphs = re.split(r'\n---+\n|\n## 第\d+场|\n##\s*第\d+场', script_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        paragraphs = [script_text]
    
    return {
        "article_script_name": "生成的剧本",
        "paragraph_num": len(paragraphs),
        "content": paragraphs
    }


@router.post("/generate", response_model=DialogueResponse)
async def generate_dialogue(request: DialogueRequest):
    try:
        save_path = tools.generate_save_path("dialogue", "dialogue")
        
        script_dict = parse_script_to_dict(request.script_content)
        
        result = dialogue(
            user_input=request.user_input,
            script=script_dict,
            save_path=save_path,
            strict_model=request.strict_model
        )
        
        results = []
        for item in result:
            results.append(DialogueItem(
                dialogues=item.get("dialogues", []),
                chapter_name=item.get("chapter_name", ""),
                site=item.get("site", "")
            ))
        
        return DialogueResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
