from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id: str):
    """Returns the video's transcript as one string, or None if it can't be fetched."""
    print(f"getting transcript for video_id: {video_id}")
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            # youtube-transcript-api 1.x
            return " ".join(snippet.text for snippet in api.fetch(video_id))
        # Older 0.x releases only have the static get_transcript()
        return " ".join(item["text"] for item in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception as e:
        print(f"an error occurred: {e}")
        return None
