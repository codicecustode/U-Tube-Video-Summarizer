from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from utils.llm_providers import LLMError, check_choice, default_choice, list_models, summarize
from utils.transcript_generate import get_transcript

load_dotenv()

app = FastAPI()


class VideoRequest(BaseModel):
    video_id: str
    provider: str | None = None
    model: str | None = None


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/models")
def models():
    choice = default_choice()
    return {
        "providers": list_models(),
        "default": {"provider": choice[0], "model": choice[1]} if choice else None,
    }


@app.post("/summarize")
def youtube_video(video_request: VideoRequest):
    if not video_request.video_id:
        return JSONResponse({"error": "video_id is required."}, status_code=400)

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
        return JSONResponse({"error": "Transcript not found or an error occurred."}, status_code=404)

    try:
        summary, truncated = summarize(provider, model, transcript)
    except LLMError as error:
        return JSONResponse({"error": str(error)}, status_code=error.status)

    print(f"Summary ({provider}/{model}): {summary}")
    return {
        "video_id": video_request.video_id,
        "provider": provider,
        "model": model,
        "truncated": truncated,
        "summarize_transcript": summary,
    }
