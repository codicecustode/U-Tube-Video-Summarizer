from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.transcript_generate import get_transcript
from utils.gemini_summarizer import summarize_text_gemini

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
    if not video_request.video_id:
       return {"error": "video_id is required."}
    try:
        transcript = get_transcript(video_request.video_id)
        #transcript = 'In the forgotten kingdom of Aerindor, hidden beyond the mountains and veiled in silver mists, there was a village called Elowen, a place untouched by time and unaware of the ancient power sleeping beneath its soil, where a boy named Kael grew up under the care of an old healer named Mira after being found as a baby at the edge of the mysterious Whispering Forest, wrapped in silk the color of stormy skies and wearing a symbol no one recognized, and though the villagers spoke in hushed voices about his strange origins and the way he always seemed to hear things others could not—the wind’s murmur, the river’s sigh, the soft laughter from within trees—they never saw the storm that was coming, nor did Kael, until one night when the moon was a perfect white coin and the air seemed thick with something ancient, he was drawn by a haunting voice calling his name, soft and melodic, leading him deep into the forest where time bent and light dimmed, until he came upon a forgotten altar covered in moss and moonlight, and there lay a pendant glowing with a pulse like a heartbeat, and the moment he touched it, his mind was flooded with visions—of a crystal city shattered, of a woman cloaked in fire screaming into silence, of a darkness spreading like spilled ink over the sky—before he collapsed, waking up with the pendant around his neck and a whisper that had followed him out of the trees, and from that day forward, Elowen changed, with animals fleeing, crops wilting, and strange dreams creeping into the minds of its people, all while Kael sought answers, finally confronting Mira who confessed the truth she had long hidden—that Kael was not just an orphan, but the child of prophecy, born when the stars aligned to hold the voice of the Lightborne, the ancient guardians who once sealed away a formless force known as the Hollow, a being of sorrow and silence that fed on forgotten memories and had once nearly devoured the world, but now stirred once again, weakened prison cracking, drawn by Kael’s awakening gift, and so Kael fled, searching for others like him, and over time he gathered a small band of kindred spirits—Aelin the blade-singer with silver in her veins, Thorn the shadow-walker who could slip through doors that weren’t there, and Lys, the girl who listened to stars and wept their secrets into the ground—and together they unraveled pieces of the lost story, traveled through abandoned temples and dreaming ruins, learning that Elowen itself was the final lock, and Kael was the key, so they returned to find the village overrun with vines that whispered and soil that trembled, and at the center stood the statue from his vision, no longer broken, its eyes lit with a cold blue flame, and Kael, knowing what he was meant to do, stepped forward not to fight but to forgive, offering not his power but his peace, his sorrow, his humanity, and in that moment, the Hollow wavered, for it had never been shown compassion, only fear, and in that silence Kael gave himself fully, and the light swallowed the darkness, and when the villagers awoke the next morning the forest was quiet and the statue was still and Kael was gone, though some say if you walk the forest on a moonlit night and hold your breath, you might still hear the Echoes, and the wind might whisper a name—Kael.'

        if transcript is None:
            return {"error": "Transcript not found or an error occurred."}
        else:
            summary = summarize_text_gemini(transcript)
            print(f"Summary: {summary}")
            return {"video_id": video_request.video_id, "summarize_transcript": summary}

    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return {"error": "An error occurred while fetching the transcript."}