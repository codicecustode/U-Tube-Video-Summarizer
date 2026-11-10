import logging
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

logger = logging.getLogger(__name__)


def get_transcript(video_id: str) -> str | None:
    """Returns the video's transcript as one string, or None if it can't be fetched."""
    logger.info("Fetching transcript for %s", video_id)

    # Try with proxy first if configured (YouTube blocks cloud IPs)
    proxy_user = os.getenv("WEBSHARE_PROXY_USER", "")
    proxy_pass = os.getenv("WEBSHARE_PROXY_PASS", "")

    try:
        if proxy_user and proxy_pass:
            proxy = WebshareProxyConfig(
                proxy_username=proxy_user,
                proxy_password=proxy_pass,
            )
            api = YouTubeTranscriptApi(proxy_config=proxy)
        else:
            api = YouTubeTranscriptApi()

        if hasattr(api, "fetch"):
            return " ".join(snippet.text for snippet in api.fetch(video_id))
        return " ".join(item["text"] for item in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception as e:
        logger.warning("Transcript fetch failed for %s: %s", video_id, e)
        return None
