"""Transcript extraction, in three steps. The first one that works wins.

Step 1: transcript API (Supadata). YouTube refuses requests from cloud server IPs,
        so on a host like Render this is the step that normally succeeds.
Step 2: yt-dlp fetches the subtitle file directly (works locally or behind a residential proxy).
Step 3: yt-dlp downloads the audio and Groq Whisper transcribes it.

Transcripts are cached in memory, so a video is only fetched once per server run.
"""
import json
import logging
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

COOKIES_PATH = os.getenv("YOUTUBE_COOKIES_PATH", "/app/cookies.txt")
PROXY_URL = os.getenv("YT_PROXY_URL", "").strip()  # http://user:pass@host:port, residential proxies only
SUPADATA_BASE_URL = os.getenv("SUPADATA_BASE_URL", "https://api.supadata.ai/v1").rstrip("/")

CACHE_SIZE = 200
_cache: "OrderedDict[str, tuple[str, str]]" = OrderedDict()


def _proxy_host() -> str:
    """The proxy's host and port for logging, without the username and password."""
    return PROXY_URL.rsplit("@", 1)[-1]


def _fetch_via_api(video_id: str) -> str | None:
    """Ask the Supadata transcript API for the video's existing captions."""
    api_key = os.getenv("SUPADATA_API_KEY", "").strip()
    if not api_key:
        logger.info("SUPADATA_API_KEY not set, skipping the transcript API")
        return None

    query = urllib.parse.urlencode({
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "text": "true",
        "mode": "native",
    })
    request = urllib.request.Request(
        f"{SUPADATA_BASE_URL}/transcript?{query}",
        headers={"x-api-key": api_key, "Accept": "application/json", "User-Agent": "yt-summarizer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            status = response.status
            data = json.loads(response.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")[:300]
        logger.warning("Transcript API returned HTTP %d for %s: %s", error.code, video_id, body)
        return None
    except Exception as error:
        logger.warning("Transcript API request failed for %s: %s", video_id, error)
        return None

    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, list):
        content = " ".join(str(chunk.get("text", "")) for chunk in content if isinstance(chunk, dict))
    if status != 200 or not isinstance(content, str) or len(content.strip()) < 50:
        logger.info("Transcript API had no usable transcript for %s (HTTP %d)", video_id, status)
        return None

    transcript = " ".join(content.split())
    logger.info("Got transcript from the API for %s: %d chars", video_id, len(transcript))
    return transcript



def _cookies_file() -> str | None:
    """Return the cookies file path if available."""
    if os.path.isfile(COOKIES_PATH):
        logger.info("Using cookies from file: %s", COOKIES_PATH)
        return COOKIES_PATH
    b64 = os.getenv("YOUTUBE_COOKIES_B64", "").strip()
    if b64:
        import base64
        tmp = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        decoded = base64.b64decode(b64)
        Path(tmp).write_bytes(decoded)
        logger.info("Decoded cookies from YOUTUBE_COOKIES_B64 (%d bytes) to %s", len(decoded), tmp)
        # Verify it looks like a valid cookies file
        first_lines = decoded.decode("utf-8", errors="ignore")[:200]
        if "youtube.com" not in first_lines.lower() and "# Netscape" not in first_lines:
            logger.warning("Cookies file doesn't look valid. First 200 chars: %s", first_lines)
        return tmp
    logger.info("No cookies configured (set YOUTUBE_COOKIES_B64 env var)")
    return None


def _clean_vtt(vtt_text: str) -> str:
    """Strip VTT timestamps, headers, and duplicate lines."""
    lines = []
    seen = set()
    for line in vtt_text.splitlines():
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line) or "-->" in line:
            continue
        if not line.strip():
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return " ".join(lines)


def _fetch_subtitles(video_id: str) -> str | None:
    """Try to get subtitles using yt-dlp (no video/audio download)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--output", os.path.join(tmp_dir, "video"),
            "--no-warnings",
            "--ignore-no-formats-error",
        ]
        cookies = _cookies_file()
        if cookies:
            cmd += ["--cookies", cookies]
        if PROXY_URL:
            cmd += ["--proxy", PROXY_URL]
            logger.info("Using proxy %s", _proxy_host())
        cmd.append(url)

        logger.info("Running yt-dlp subtitle cmd: %s", " ".join(cmd[:8]) + " ...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.warning("yt-dlp subtitle failed (exit %d). stderr: %s", result.returncode, result.stderr[-500:])
                logger.warning("yt-dlp stdout: %s", result.stdout[-500:])
                return None
            # Log success info
            logger.info("yt-dlp subtitle succeeded. stdout tail: %s", result.stdout[-300:])
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp subtitle timed out for %s", video_id)
            return None

        vtt_files = list(Path(tmp_dir).glob("*.vtt"))
        if not vtt_files:
            logger.info("No .vtt files found in %s. Files present: %s", tmp_dir, list(Path(tmp_dir).iterdir()))
            return None

        vtt_text = vtt_files[0].read_text(encoding="utf-8", errors="ignore")
        transcript = _clean_vtt(vtt_text)
        if len(transcript) < 50:
            logger.info("Subtitle text too short for %s (%d chars)", video_id, len(transcript))
            return None

        logger.info("Got subtitles for %s: %d chars", video_id, len(transcript))
        return transcript


def _fetch_audio_transcript(video_id: str) -> str | None:
    """Download audio with yt-dlp, transcribe with Groq Whisper."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        logger.warning("GROQ_API_KEY not set, cannot use Whisper fallback")
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = os.path.join(tmp_dir, "audio.m4a")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "m4a",
            "--audio-quality", "worst",
            "--output", audio_path,
            "--no-warnings",
            "--max-filesize", "25M",
        ]
        cookies = _cookies_file()
        if cookies:
            cmd += ["--cookies", cookies]
        if PROXY_URL:
            cmd += ["--proxy", PROXY_URL]
        cmd.append(url)

        logger.info("Running yt-dlp audio cmd for %s", video_id)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning("yt-dlp audio failed (exit %d): %s", result.returncode, result.stderr[-500:])
                return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp audio timed out for %s", video_id)
            return None

        audio_files = list(Path(tmp_dir).glob("audio.*"))
        if not audio_files:
            logger.warning("No audio file found for %s", video_id)
            return None
        actual_path = audio_files[0]

        size_mb = actual_path.stat().st_size / (1024 * 1024)
        if size_mb > 25:
            logger.warning("Audio too large for Whisper: %.1f MB", size_mb)
            return None

        logger.info("Transcribing audio for %s (%.1f MB)", video_id, size_mb)

        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            with open(actual_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    response_format="text",
                )
            transcript = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
            if len(transcript) < 50:
                logger.info("Whisper transcript too short: %d chars", len(transcript))
                return None
            logger.info("Whisper transcript for %s: %d chars", video_id, len(transcript))
            return transcript
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return None


def get_transcript(video_id: str) -> tuple[str | None, str]:
    """Get a video's transcript. Returns (text, method): method is 'api', 'subtitles', 'whisper' or 'failed'."""
    if video_id in _cache:
        _cache.move_to_end(video_id)
        logger.info("Transcript for %s served from cache", video_id)
        return _cache[video_id]

    logger.info("=== Fetching transcript for %s ===", video_id)
    for method, fetch in (("api", _fetch_via_api), ("subtitles", _fetch_subtitles), ("whisper", _fetch_audio_transcript)):
        text = fetch(video_id)
        if text:
            _cache[video_id] = (text, method)
            if len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
            return text, method
        logger.info("Step '%s' gave no transcript for %s", method, video_id)

    return None, "failed"
