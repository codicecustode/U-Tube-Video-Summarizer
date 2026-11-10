"""Transcript extraction: yt-dlp subtitles first, Groq Whisper audio fallback.

Path 1 (95% of videos): yt-dlp fetches auto/manual subtitles → parse VTT → clean text
Path 2 (fallback):       yt-dlp downloads audio only → Groq Whisper transcribes → text
"""
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Cookies support ──────────────────────────────────────────────────────────
# Export cookies from your browser (e.g. with "Get cookies.txt" extension),
# then either mount the file at /app/cookies.txt or base64-encode it into
# the YOUTUBE_COOKIES_B64 env var.
COOKIES_PATH = os.getenv("YOUTUBE_COOKIES_PATH", "/app/cookies.txt")


def _cookies_file() -> str | None:
    """Return the cookies file path if available."""
    # From direct file
    if os.path.isfile(COOKIES_PATH):
        return COOKIES_PATH
    # From base64 env var (for Render / cloud deploys)
    b64 = os.getenv("YOUTUBE_COOKIES_B64", "").strip()
    if b64:
        import base64
        tmp = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        Path(tmp).write_bytes(base64.b64decode(b64))
        return tmp
    return None


# ── Path 1: Subtitles via yt-dlp ─────────────────────────────────────────────
def _clean_vtt(vtt_text: str) -> str:
    """Strip VTT timestamps, headers, and duplicate lines."""
    lines = []
    seen = set()
    for line in vtt_text.splitlines():
        # Skip VTT headers, timestamps, and blank lines
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line) or "-->" in line:
            continue
        if not line.strip():
            continue
        # Remove HTML tags like <c> </c>
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
            "--output", os.path.join(tmp_dir, "%(id)s"),
            "--no-warnings",
            "--quiet",
        ]
        cookies = _cookies_file()
        if cookies:
            cmd += ["--cookies", cookies]
        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning("yt-dlp subtitle fetch failed: %s", result.stderr[:200])
                return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp subtitle fetch timed out for %s", video_id)
            return None

        # Look for the downloaded VTT file
        vtt_files = list(Path(tmp_dir).glob("*.vtt"))
        if not vtt_files:
            logger.info("No subtitle files found for %s", video_id)
            return None

        vtt_text = vtt_files[0].read_text(encoding="utf-8", errors="ignore")
        transcript = _clean_vtt(vtt_text)
        if len(transcript) < 50:
            logger.info("Subtitle text too short for %s (%d chars)", video_id, len(transcript))
            return None

        logger.info("Got subtitles for %s: %d chars", video_id, len(transcript))
        return transcript


# ── Path 2: Audio + Whisper fallback ──────────────────────────────────────────
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
            "--audio-quality", "worst",  # smallest file
            "--output", audio_path,
            "--no-warnings",
            "--quiet",
            "--max-filesize", "25M",  # Whisper limit
        ]
        cookies = _cookies_file()
        if cookies:
            cmd += ["--cookies", cookies]
        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning("yt-dlp audio download failed: %s", result.stderr[:200])
                return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp audio download timed out for %s", video_id)
            return None

        # Find the actual audio file (yt-dlp may add extension)
        audio_files = list(Path(tmp_dir).glob("audio.*"))
        if not audio_files:
            logger.warning("No audio file found for %s", video_id)
            return None
        actual_path = audio_files[0]

        # Check file size
        size_mb = actual_path.stat().st_size / (1024 * 1024)
        if size_mb > 25:
            logger.warning("Audio too large for Whisper: %.1f MB", size_mb)
            return None

        logger.info("Transcribing audio for %s (%.1f MB)", video_id, size_mb)

        # Send to Groq Whisper
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


# ── Public API ────────────────────────────────────────────────────────────────
def get_transcript(video_id: str) -> tuple[str | None, str]:
    """Get transcript for a video. Returns (text, method) where method is
    'subtitles', 'whisper', or 'failed'."""
    logger.info("Fetching transcript for %s", video_id)

    # Try subtitles first (fast, free, no API key needed)
    text = _fetch_subtitles(video_id)
    if text:
        return text, "subtitles"

    # Fallback to audio + Whisper
    logger.info("No subtitles, trying Whisper fallback for %s", video_id)
    text = _fetch_audio_transcript(video_id)
    if text:
        return text, "whisper"

    return None, "failed"
