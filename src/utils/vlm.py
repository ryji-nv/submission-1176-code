"""VLM query utility for image-based judging (external VLM service)."""

import base64
import json
import os

import requests

API_BASE = os.environ.get("VLM_API_BASE", "https://example.com/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "Qwen3-VL-235B-A22B-Thinking")


def encode_image(image_path: str) -> str:
    """Return a base64 data URL for a JPEG image."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def call_vlm(
    messages: list[dict],
    model: str = MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    seed: int = 1,
    timeout: int = 60,
) -> tuple[str, str]:
    """
    Send messages to the VLM and return (content, thinking).

    Raises:
        RuntimeError: On non-200 response or empty content.
    """
    key = os.environ.get("VLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("VLM_API_KEY is not set.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
    }

    max_attempts = 3
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                API_BASE, data=json.dumps(payload), headers=headers, timeout=timeout
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"VLM API error {response.status_code}: {response.text}"
                )

            result = response.json()
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                print(f"WARNING: VLM response truncated (hit max_tokens={max_tokens})")

            msg = choice["message"]
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", p.get("content", ""))
                    for p in content
                    if isinstance(p, dict)
                )
            content = (content or "").strip()
            if not content:
                raise RuntimeError("VLM returned empty content.")
            thinking = (msg.get("reasoning") or "").strip()
            return content, thinking
        except Exception as e:
            last_exc = e
            if attempt < max_attempts:
                print(f"WARNING: VLM attempt {attempt} failed ({e}), retrying...")
    raise last_exc
