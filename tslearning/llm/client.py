"""
Provider-agnostic LLM client — one `complete()` call, several back ends.

Nothing here knows anything about rules. It takes a prompt and returns text, so the
experiment can be run against a local model, a hosted one, or a canned file without any
other code changing.

────────────────────────────────────────────────────────────────────────────────
CHOOSING A PROVIDER
────────────────────────────────────────────────────────────────────────────────

  ollama     LOCAL, free, no data leaves the machine, and — the part that matters for a
             paper — still runnable in two years. `ollama serve`, then e.g.
             `ollama pull llama3.1`. Recommended for the reported experiment.
  openai     hosted; needs OPENAI_API_KEY. Record the exact model string AND the date:
             hosted model names are reassigned, so "gpt-4o" in 2026 is not the "gpt-4o"
             a reader will get later. That is a reproducibility problem, not a nitpick.
  anthropic  hosted; needs ANTHROPIC_API_KEY. Same caveat.
  file       reads a saved completion from disk. For developing the parser and the
             scoring path with no network at all, and for re-running the exact
             experiment from an archived response.

Before using a hosted provider, check your institution's policy on outbound API calls.
The inputs here are public market prices, but a blanket rule may still apply.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_MODELS = {
    "ollama": "llama3.1",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}


class LLMError(RuntimeError):
    pass


# Rough chars-per-token for English + numeric tables. Only used to warn, never to decide.
CHARS_PER_TOKEN = 4.0
# Ollama's own default is 4096, which SILENTLY TRUNCATES a prompt of this size — the model
# would answer from a fragment and nothing in the response would say so. Llama 3.1 and most
# current local models handle 128k, so ask for a context that fits the whole prompt.
DEFAULT_NUM_CTX = 32768


def complete(prompt: str, provider: str = "ollama", model: "str | None" = None,
             temperature: float = 0.0, max_tokens: int = 4096,
             num_ctx: int = DEFAULT_NUM_CTX, json_mode: bool = True, path=None,
             timeout: float = 600.0, verbose: bool = True) -> dict:
    """Send `prompt`, return {"text", "provider", "model", "seconds", "raw"}.

    `temperature=0` by default: this is a measurement, and a rule set that changes between
    runs cannot be compared against a deterministic template fitter. Report the
    temperature in the paper either way.

    `num_ctx` is passed to ollama explicitly. Relying on the server's default is how a
    14,000-token prompt gets quietly cut to 4,096 and the experiment measures a model
    reasoning about the first fifth of the data.
    """
    model = model or DEFAULT_MODELS.get(provider, "")
    t0 = time.perf_counter()

    est = int(len(prompt) / CHARS_PER_TOKEN)
    if verbose:
        print(f"prompt ~{est:,} tokens ({len(prompt):,} chars)")
    if provider == "ollama" and est > num_ctx * 0.9:
        raise LLMError(
            f"the prompt is ~{est:,} tokens but num_ctx is {num_ctx:,}. Ollama would "
            f"TRUNCATE it silently and the model would answer from a fragment.\n"
            f"Either raise it (--num-ctx {max(32768, 1 << (est * 2 - 1).bit_length())}) "
            f"or shorten the input (--max-days), and note that a larger context costs "
            f"memory for the KV cache.")

    if provider == "file":
        if path is None:
            raise LLMError("provider 'file' needs path=<saved completion>")
        text, raw = Path(path).read_text(), None
    elif provider == "ollama":
        text, raw = _ollama(prompt, model, temperature, max_tokens, timeout, num_ctx,
                            json_mode)
    elif provider == "openai":
        text, raw = _openai(prompt, model, temperature, max_tokens, timeout)
    elif provider == "anthropic":
        text, raw = _anthropic(prompt, model, temperature, max_tokens, timeout)
    else:
        raise LLMError(f"unknown provider {provider!r}; "
                       f"expected one of ollama|openai|anthropic|file")

    dt = time.perf_counter() - t0
    if isinstance(raw, dict) and "_temperature_used" in raw:
        temperature = raw["_temperature_used"]       # what was actually sent
    if verbose:
        print(f"{provider}/{model}: {len(text)} chars in {dt:.1f}s")
    return {"text": text, "provider": provider, "model": model,
            "seconds": dt, "temperature": temperature, "raw": raw}


# ── back ends ─────────────────────────────────────────────────────────────────

def _ollama(prompt, model, temperature, max_tokens, timeout, num_ctx=DEFAULT_NUM_CTX,
            json_mode: bool = True):
    """`json_mode` uses ollama's constrained decoding so the reply is valid JSON.

    On by default because a completion that isn't JSON is a hard failure for the parser,
    and a small local model will happily answer a different question — llama3.1:8b replied
    to its first attempt with Python implementations of the predicates.

    Methodological note: constraining the format means "could it follow the output
    contract?" is no longer being measured. That unconstrained failure IS reportable, so
    `--no-json-mode` keeps it available; the mode used is recorded in the run metadata
    either way.
    """
    import requests
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": num_ctx}}
    if json_mode:
        body["format"] = "json"
    try:
        r = requests.post(f"{host}/api/generate", timeout=timeout, json=body)
    except Exception as e:
        raise LLMError(f"cannot reach ollama at {host} — is `ollama serve` running? ({e})")
    if r.status_code == 404:
        raise LLMError(f"ollama does not have model {model!r} — `ollama pull {model}`")
    if r.status_code != 200:
        raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    # prompt_eval_count is how many tokens the model ACTUALLY read. If it is far below the
    # estimate, the prompt was truncated regardless of what we asked for.
    n_in = d.get("prompt_eval_count")
    if n_in and n_in < len(prompt) / CHARS_PER_TOKEN * 0.7:
        raise LLMError(
            f"ollama read only {n_in:,} prompt tokens for a ~{int(len(prompt)/CHARS_PER_TOKEN):,}"
            f"-token prompt — it was TRUNCATED. The model did not see most of the data, so "
            f"the result is not usable. Raise --num-ctx (currently {num_ctx:,}) or lower "
            f"--max-days.")
    return d.get("response", ""), d


def _openai(prompt, model, temperature, max_tokens, timeout):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is not set")
    import requests
    r = requests.post("https://api.openai.com/v1/chat/completions", timeout=timeout,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": temperature,
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise LLMError(f"openai HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    return d["choices"][0]["message"]["content"], d


def _anthropic(prompt, model, temperature, max_tokens, timeout):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set")
    import requests

    def _post(with_temperature: bool):
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if with_temperature and temperature is not None:
            body["temperature"] = temperature
        return requests.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                      "content-type": "application/json"},
                             json=body)

    r = _post(True)
    used = temperature
    # Newer models reject `temperature` ("deprecated for this model"): retry without it
    # and RECORD that the model's own default sampling was used — the run metadata must
    # not claim a temperature that was never applied.
    if r.status_code == 400 and "temperature" in r.text:
        print(f"note: {model} does not accept `temperature`; using the model's default "
              f"sampling (recorded as temperature=None)")
        r, used = _post(False), None
    if r.status_code != 200:
        raise LLMError(f"anthropic HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    d["_temperature_used"] = used
    return "".join(b.get("text", "") for b in d.get("content", [])), d


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a completion.

    Models wrap JSON in prose or fences however they like, so this is deliberately
    forgiving — but it does NOT repair malformed JSON. A model that cannot emit valid
    JSON when asked is a result worth recording, not something to paper over.
    """
    s = text.strip()
    if "```" in s:                                   # fenced block
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise LLMError(f"no JSON object found in the completion. First 300 chars:\n"
                       f"{text[:300]}")
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"the completion contained malformed JSON ({e}). "
                       f"Record this — it is an outcome, not a bug to work around.")
