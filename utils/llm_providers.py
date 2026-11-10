"""LLM providers the summarizer can use.

Every provider below accepts the OpenAI chat-completions format, so one client
works for all of them: only the base URL, the API key and the model name change.
A provider becomes available as soon as its API key is set in the environment.

Long transcripts are split into chunks, summarized in parallel, then combined.
"""
import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "max_input_chars": 400_000,
        "models": {
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        },
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "max_input_chars": 20_000,
        "models": {
            "openai/gpt-oss-20b": "GPT-OSS 20B",
            "openai/gpt-oss-120b": "GPT-OSS 120B",
            "qwen/qwen3.8-27b": "Qwen3.8 27B",
        },
    },
    "nvidia": {
        "label": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "max_input_chars": 100_000,
        "models": {
            "deepseek-ai/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
            "nvidia/nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning 30B",
            "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
        },
    },
}

DEFAULT_CHOICE = ("gemini", "gemini-2.5-flash")

# ── Chunk size for splitting long transcripts ─────────────────────────────────
# Each chunk gets its own summarize call; all chunks run in parallel.
CHUNK_SIZE = 15_000  # chars per chunk (safe for all providers)

PROMPT = (
    "You are an intelligent summarizer. Given a YouTube video transcript, generate a clear, visually appealing summary in bullet points format.\n\n"
    "Requirements:\n"
    "- Use only the most important insights, facts, or key takeaways\n"
    "- Start each bullet with a fitting emoji like 💡 (idea), 🎯 (key point), 📌 (fact), or 🚀 (tip) based on the content\n"
    "- Add one blank line between each bullet point for better readability\n"
    "- Use simple, clear language that's easy to scan and understand\n"
    "- Do NOT include unimportant information, repetition, or long paragraphs\n\n"
    "Here is the YouTube video transcript:\n\n"
)

CHUNK_PROMPT = (
    "You are an intelligent summarizer. This is PART {part_num} of {total_parts} of a YouTube video transcript. "
    "Summarize only the key points from this part in bullet points. "
    "Start each bullet with a fitting emoji. Be concise.\n\n"
    "Transcript part:\n\n"
)

COMBINE_PROMPT = (
    "Below are summaries of different parts of the same YouTube video. "
    "Combine them into one final, coherent summary.\n\n"
    "Requirements:\n"
    "- Merge duplicate or overlapping points\n"
    "- Keep the most important insights from all parts\n"
    "- Start each bullet with a fitting emoji\n"
    "- Add one blank line between each bullet point\n"
    "- The final summary should read as one unified document, not separate parts\n\n"
    "Part summaries:\n\n"
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Reuse one client per provider
_clients: dict[str, OpenAI] = {}
_executor = ThreadPoolExecutor(max_workers=6)


def _get_client(provider: str) -> OpenAI:
    if provider not in _clients:
        config = PROVIDERS[provider]
        _clients[provider] = OpenAI(
            base_url=config["base_url"],
            api_key=_api_key(provider),
            timeout=120,
            max_retries=1,
        )
    return _clients[provider]


class LLMError(Exception):
    """An error that is safe to show to the user, with the HTTP status to return."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _api_key(provider: str) -> str:
    return os.getenv(PROVIDERS[provider]["key_env"], "").strip()


def list_models() -> list[dict]:
    """Providers and their models for the frontend, flagging which ones have a key set."""
    return [
        {
            "id": provider_id,
            "label": provider["label"],
            "available": bool(_api_key(provider_id)),
            "models": [{"id": model_id, "label": label} for model_id, label in provider["models"].items()],
        }
        for provider_id, provider in PROVIDERS.items()
    ]


def default_choice() -> tuple[str, str] | None:
    """Gemini 2.5 Flash if its key is set, otherwise the first model of any provider with a key."""
    if _api_key(DEFAULT_CHOICE[0]):
        return DEFAULT_CHOICE
    for provider_id, provider in PROVIDERS.items():
        if _api_key(provider_id):
            return provider_id, next(iter(provider["models"]))
    return None


def check_choice(provider: str, model: str) -> None:
    """Rejects providers and models that aren't configured here."""
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}", 400)
    if model not in PROVIDERS[provider]["models"]:
        raise LLMError(f"{PROVIDERS[provider]['label']} has no model called {model} in this app.", 400)
    if not _api_key(provider):
        key_env = PROVIDERS[provider]["key_env"]
        raise LLMError(f"{PROVIDERS[provider]['label']} isn't set up yet: add {key_env} to your environment and restart.", 503)


def _call_llm(provider: str, model: str, prompt: str) -> str:
    """Single LLM call. Returns the cleaned response text."""
    config = PROVIDERS[provider]
    model_label = config["models"][model]
    client = _get_client(provider)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except openai.RateLimitError:
        raise LLMError(f"{model_label} has reached its free usage limit for now. Pick another model or try again in a minute.", 429)
    except openai.AuthenticationError:
        raise LLMError(f"{config['label']} rejected the API key in {config['key_env']}. Check that it's correct.", 502)
    except openai.NotFoundError:
        raise LLMError(f"{config['label']} doesn't offer {model_label} anymore. Pick another model.", 502)
    except (openai.APIConnectionError, openai.APITimeoutError):
        raise LLMError(f"Couldn't reach {config['label']}. Try again in a moment.", 504)
    except openai.APIStatusError as error:
        if error.status_code == 413:
            raise LLMError(f"This video is too long for {model_label}'s free limit. Try a Gemini model.", 413)
        raise LLMError(f"{config['label']} returned an error ({error.status_code}). Try another model.", 502)

    text = THINK_BLOCK.sub("", response.choices[0].message.content or "").strip()
    if not text:
        raise LLMError(f"{model_label} returned an empty reply. Try another model.", 502)
    return text


def _split_transcript(transcript: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split transcript into chunks, breaking at sentence boundaries where possible."""
    if len(transcript) <= chunk_size:
        return [transcript]

    chunks = []
    start = 0
    while start < len(transcript):
        end = start + chunk_size
        if end >= len(transcript):
            chunks.append(transcript[start:])
            break
        # Try to break at a sentence boundary (. ! ? followed by space)
        break_point = transcript.rfind(". ", start + chunk_size - 2000, end)
        if break_point == -1:
            break_point = transcript.rfind(" ", start + chunk_size - 500, end)
        if break_point == -1:
            break_point = end
        else:
            break_point += 1  # include the period/space
        chunks.append(transcript[start:break_point])
        start = break_point

    return chunks


def summarize(provider: str, model: str, transcript: str) -> tuple[str, bool]:
    """Summarizes a transcript. Short ones go direct, long ones are chunked in parallel."""
    check_choice(provider, model)

    # Short transcript: single call (same as before)
    if len(transcript) <= CHUNK_SIZE:
        logger.info("Short transcript (%d chars), single call", len(transcript))
        summary = _call_llm(provider, model, PROMPT + transcript)
        return summary, False

    # Long transcript: chunk → parallel summarize → combine
    chunks = _split_transcript(transcript)
    logger.info("Long transcript (%d chars), split into %d chunks", len(transcript), len(chunks))

    # Summarize each chunk (runs in thread pool for true parallelism with sync OpenAI client)
    def summarize_chunk(i: int, chunk: str) -> str:
        prompt = CHUNK_PROMPT.format(part_num=i + 1, total_parts=len(chunks)) + chunk
        return _call_llm(provider, model, prompt)

    loop = asyncio.new_event_loop()
    try:
        futures = [
            loop.run_in_executor(_executor, summarize_chunk, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        chunk_summaries = loop.run_until_complete(asyncio.gather(*futures))
    finally:
        loop.close()

    logger.info("All %d chunks summarized, combining", len(chunks))

    # Combine all chunk summaries into one final summary
    combined_input = ""
    for i, cs in enumerate(chunk_summaries):
        combined_input += f"--- Part {i + 1} of {len(chunks)} ---\n{cs}\n\n"

    final_summary = _call_llm(provider, model, COMBINE_PROMPT + combined_input)
    return final_summary, False


async def async_summarize(provider: str, model: str, transcript: str) -> tuple[str, bool]:
    """Async wrapper for use in FastAPI async routes."""
    return await asyncio.to_thread(summarize, provider, model, transcript)
