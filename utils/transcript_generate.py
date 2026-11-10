import logging

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


def get_transcript(video_id: str) -> str | None:
    """Returns the video's transcript as one string, or None if it can't be fetched."""
    logger.info("Fetching transcript for %s", video_id)
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            return " ".join(snippet.text for snippet in api.fetch(video_id))
        return " ".join(item["text"] for item in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception as e:
        logger.warning("Transcript fetch failed for %s: %s", video_id, e)
        return None
