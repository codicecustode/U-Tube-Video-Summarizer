"""LLM providers the summarizer can use.

Every provider below accepts the OpenAI chat-completions format, so one client
works for all of them: only the base URL, the API key and the model name change.
A provider becomes available as soon as its API key is set in the environment.
"""
import logging
import os
import re

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

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Reuse one client per provider instead of creating a new one every request
_clients: dict[str, OpenAI] = {}


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
    """Rejects providers and models that aren't configured here, so the API keys can't be used for anything else."""
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider: {provider}", 400)
    if model not in PROVIDERS[provider]["models"]:
        raise LLMError(f"{PROVIDERS[provider]['label']} has no model called {model} in this app.", 400)
    if not _api_key(provider):
        key_env = PROVIDERS[provider]["key_env"]
        raise LLMError(f"{PROVIDERS[provider]['label']} isn't set up yet: add {key_env} to your environment and restart.", 503)


def summarize(provider: str, model: str, transcript: str) -> tuple[str, bool]:
    """Summarizes a transcript with the chosen model. Returns the summary and whether the transcript was shortened."""
    check_choice(provider, model)
    config = PROVIDERS[provider]
    model_label = config["models"][model]

    truncated = len(transcript) > config["max_input_chars"]
    if truncated:
        transcript = transcript[: config["max_input_chars"]]

    client = _get_client(provider)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT + transcript}],
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

    summary = THINK_BLOCK.sub("", response.choices[0].message.content or "").strip()
    if not summary:
        raise LLMError(f"{model_label} returned an empty reply. Try another model.", 502)
    return summary, truncated
