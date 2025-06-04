from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.transcript_generate import get_transcript
from utils.summarizer import summarize_text

import torch

app = FastAPI()

class VideoRequest(BaseModel):
    video_id: str

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    print("gpu available: ", torch.cuda.is_available())
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
def youtube_video(video_request: VideoRequest):
    #if not video_request.video_id:
    #    return {"error": "video_id is required."}
    try:
        transcript = get_transcript(video_request.video_id)

        if transcript is None:
            print(f"Transcript for video_id {video_request.video_id} not found or an error occurred.")
            return {"error": "Transcript not found or an error occurred."}
        else:
            print(f"Transcript for video_id {video_request.video_id} retrieved successfully.")
            # Summarize the transcript
            summary = summarize_text(transcript)
            print(f"Summary for video_id {video_request.video_id} generated successfully.")
            print(f"Summary: {summary}")

            return {"video_id": video_request.video_id, "summarize_transcript": summary}

    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return {"error": "An error occurred while fetching the transcript."}