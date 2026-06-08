"""Sonnet-as-judge for grading WhatsApp reply tone/correctness.

Uses the same model the agent itself uses (claude-sonnet-4-6) so the judge has
matching priors about Pakistani B2B WhatsApp register. Cheap (~200 tokens).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
JUDGE_SYSTEM = (
    "You grade WhatsApp replies for an Umrah B2B agent that talks to Pakistani "
    "travel agencies. Apply the criterion strictly. The agent must sound like a "
    "human teammate, not a corporate bot. No em dashes (—). No emojis. "
    "Hyphens INSIDE UB numbers (UB-NNNNNN format) are correct and required — do not penalize them. "
    "No 'saved X of Y' progress math. No 'flagged for review'. Match the user's "
    "language. Output ONLY a JSON object on a single line with shape: "
    '{"pass": <bool>, "score": <float 0-1>, "reason": "<short>"}'
)


@dataclass
class JudgeResult:
    passed: bool
    score: float
    reason: str
    cost_usd: float = 0.0


def _client() -> Any:
    if anthropic is None:
        raise RuntimeError("anthropic SDK not installed; pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


_PRICING = {
    # rough per-MTok in USD for sonnet 4.6
    "input": 3.0 / 1_000_000,
    "output": 15.0 / 1_000_000,
}


def grade(criterion: str, reply_text: str) -> JudgeResult:
    """Run the judge over a single reply. Returns pass/fail + score + reason."""
    if not reply_text:
        return JudgeResult(False, 0.0, "empty reply", 0.0)
    client = _client()
    user = (
        f"CRITERION:\n{criterion}\n\n"
        f"REPLY (verbatim, between <<< >>>):\n<<<{reply_text}>>>\n\n"
        "Grade this reply."
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    parsed = _parse_json(text)
    cost = (
        getattr(resp.usage, "input_tokens", 0) * _PRICING["input"]
        + getattr(resp.usage, "output_tokens", 0) * _PRICING["output"]
    )
    if parsed is None:
        return JudgeResult(False, 0.0, f"judge returned non-JSON: {text[:120]}", cost)
    return JudgeResult(
        passed=bool(parsed.get("pass")),
        score=float(parsed.get("score") or 0.0),
        reason=str(parsed.get("reason") or "")[:300],
        cost_usd=cost,
    )


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (TypeError, ValueError):
        return None
