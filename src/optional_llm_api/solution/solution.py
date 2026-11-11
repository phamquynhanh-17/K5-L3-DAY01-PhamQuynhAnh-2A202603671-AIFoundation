"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Solution for optional side lab.
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD)
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Call the OpenAI Chat Completions API and return the response text + latency."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    latency = max(time.perf_counter() - t0, 0.001)
    text = resp.choices[0].message.content or ""
    return text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Call the OpenAI Chat Completions API using OPENAI_MINI_MODEL."""
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """Call both models and return a comparison dictionary."""
    gpt4o_text, gpt4o_lat = call_openai(prompt)
    mini_text, mini_lat = call_openai_mini(prompt)

    est_tokens = len(gpt4o_text.split()) / 0.75
    cost = (est_tokens / 1000.0) * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]

    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_lat,
        "mini_latency": mini_lat,
        "gpt4o_cost_estimate": cost,
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Run an interactive streaming chatbot in the terminal."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=history,
            stream=True,
            reasoning_effort="none",
            max_completion_tokens=256,
        )
        assistant_reply = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_reply.append(delta)
            print(delta, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": "".join(assistant_reply)})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Call fn() with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """Run compare_models on each prompt in the list."""
    results = []
    for p in prompts:
        res = compare_models(p)
        res["prompt"] = p
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """Format a list of compare_models results as a readable text table."""
    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for r in results:
        p = (r.get("prompt", "")[:37] + "...") if len(r.get("prompt", "")) > 40 else r.get("prompt", "")
        g = (r.get("gpt4o_response", "")[:37] + "...") if len(r.get("gpt4o_response", "")) > 40 else r.get("gpt4o_response", "")
        m = (r.get("mini_response", "")[:37] + "...") if len(r.get("mini_response", "")) > 40 else r.get("mini_response", "")
        glat = f"{r.get('gpt4o_latency', 0.0):.2f}s"
        mlat = f"{r.get('mini_latency', 0.0):.2f}s"
        lines.append(f"{p} | {g} | {m} | {glat} | {mlat}")
    return "\n".join(lines)


if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

