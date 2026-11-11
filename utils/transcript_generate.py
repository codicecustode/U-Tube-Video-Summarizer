"""Transcript extraction: yt-dlp subtitles first, Groq Whisper audio fallback.

Path 1 (95% of videos): yt-dlp fetches auto/manual subtitles -> parse VTT -> clean text
Path 2 (fallback):       yt-dlp downloads audio only -> Groq Whisper transcribes -> text
"""
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

COOKIES_PATH = os.getenv("YOUTUBE_COOKIES_PATH", "/app/cookies.txt")


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
            "--verbose",
        ]
        cookies = _cookies_file()
        if cookies:
            cmd += ["--cookies", cookies]
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
    """Get transcript for a video. Returns (text, method)."""
    logger.info("=== Fetching transcript for %s ===", video_id)

    text = _fetch_subtitles(video_id)
    if text:
        return text, "subtitles"

    logger.info("No subtitles, trying Whisper fallback for %s", video_id)
    text = _fetch_audio_transcript(video_id)
    if text:
        return text, "whisper"

    return None, "failed"
