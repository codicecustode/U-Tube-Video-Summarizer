from fastapi import FastAPI
from pydantic import BaseModel
from utils.transcript_generate import get_transcript
from utils.summarizer import summarize_text
app = FastAPI()

class VideoRequest(BaseModel):
    video_id: str

@app.get("/")
def root_page():
    return {"message": "Hello World"}

@app.get("/youtube-video")
def youtube_video():
    #if not video_request.video_id:
    #    return {"error": "video_id is required."}
    try:
        transcript = get_transcript('kY14KfZQ1TI')
    
        if transcript is None:
            print(f"Transcript for video_id {"video_request.video_id"} not found or an error occurred.")
            return {"error": "Transcript not found or an error occurred."}
        else:
            print(f"Transcript for video_id {"video_request.video_id"} retrieved successfully.")
            print(f"Transcript: {transcript}")
            # Summarize the transcript
            summary = summarize_text(transcript)
        
            return {"video_id": "video_request.video_id", "summarize_transcript": summary}

    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return {"error": "An error occurred while fetching the transcript."}