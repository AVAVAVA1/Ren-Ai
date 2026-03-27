from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import outline, script, dialogue, story, runninghub, play

app = FastAPI(title="Story Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(outline.router)
app.include_router(script.router)
app.include_router(dialogue.router)
app.include_router(story.router)
app.include_router(runninghub.router)
app.include_router(play.router)


@app.get("/")
async def root():
    return {"message": "Story Generation API is running"}
