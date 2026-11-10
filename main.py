import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

from utils.llm_providers import LLMError, check_choice, default_choice, list_models, summarize
from utils.transcript_generate import get_transcript

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Video Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class VideoRequest(BaseModel):
    video_id: str
    provider: str | None = None
    model: str | None = None

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) != 11:
            raise ValueError("video_id must be an 11-character YouTube video ID")
        return v


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root_page(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/models")
async def models():
    choice = default_choice()
    return {
        "providers": list_models(),
        "default": {"provider": choice[0], "model": choice[1]} if choice else None,
    }


@app.post("/summarize")
async def youtube_video(video_request: VideoRequest):
    provider, model = video_request.provider, video_request.model
    if not provider or not model:
        choice = default_choice()
        if choice is None:
            return JSONResponse(
                {"error": "No API keys are set yet. Add GEMINI_API_KEY, GROQ_API_KEY or NVIDIA_API_KEY and restart."},
                status_code=503,
            )
        provider, model = choice

    try:
        check_choice(provider, model)
    except LLMError as error:
        return JSONResponse({"error": str(error)}, status_code=error.status)

    transcript = get_transcript(video_request.video_id)
    if transcript is None:
        return JSONResponse({"error": "Could not fetch the transcript. The video may not have captions enabled."}, status_code=404)

    try:
        summary, truncated = summarize(provider, model, transcript)
    except LLMError as error:
        return JSONResponse({"error": str(error)}, status_code=error.status)

    logger.info("Summarized %s with %s/%s (%d chars transcript)", video_request.video_id, provider, model, len(transcript))
    return {
        "video_id": video_request.video_id,
        "provider": provider,
        "model": model,
        "truncated": truncated,
        "summary": summary,
    }
