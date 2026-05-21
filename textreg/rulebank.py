"""
RuleBank R_t: cross-step memory of accepted generalized rules.

Tracks, for each canonical rule description r, the mention count m_t(r) --
the number of times r has matched an accepted purified gradient up through
step t. The recurrence frequency m_t(r) is used by Stage 1 (Dual-Evidence
Gradient Purification) and Stage 2 (M_Delta) as a monotone empirical proxy
for the rule's true scope s(r):

    \\hat{s}_t(r) = psi(m_t(r)),    psi' >= 0.

Each accepted gradient is first canonicalized by an LLM-based matcher
(canonicalize-then-match) and either merged into an existing entry r
(m_t(r) = m_{t-1}(r) + 1) or inserted as a new canonical description r*
(m_t(r*) = 1). Aggregating semantically equivalent rules under one entry
avoids fragmenting counts across surface variants.

Because R_t is updated only with accepted gradients, m_t(r) counts how
often a rule has survived purification rather than how often it appears in
raw model outputs, which prevents noisy or batch-specific feedback from
contaminating the recurrence statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Rule:
    rule_id: str
    canonical_description: str
    mention_count: int = 0


class RuleBank:
    """
    R_t = {(r, m_t(r))}: maps rule_id -> Rule. Entries accumulate across
    optimization steps and epochs; m_t(r) is `Rule.mention_count`.
    """

    def __init__(self) -> None:
        self.rules: dict[str, Rule] = {}
        self.next_id: int = 1

    def insert(self, canonical_description: str, count: int = 1) -> str:
        rule_id = f"R{self.next_id}"
        self.next_id += 1
        self.rules[rule_id] = Rule(
            rule_id=rule_id,
            canonical_description=canonical_description,
            mention_count=count,
        )
        return rule_id

    def increment(self, rule_id: str, value: int = 1) -> None:
        if rule_id in self.rules:
            self.rules[rule_id].mention_count += value

    def get_summary(self, max_rules: int = 30) -> str:
        """Render the top-N rules by mention_count for inclusion in LLM prompts."""
        if not self.rules:
            return "(empty)"
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.mention_count,
            reverse=True,
        )[:max_rules]
        return "\n".join(
            f"{r.rule_id} (count={r.mention_count}): {r.canonical_description}"
            for r in sorted_rules
        )

    def apply_operations(self, operations: list[dict]) -> None:
        """Apply a batch of {type: insert|increment, ...} ops produced by an LLM."""
        for op in operations:
            op_type = op.get("type", "")
            if op_type == "increment":
                self.increment(op.get("rule_id", ""), int(op.get("value", 1)))
            elif op_type == "insert":
                desc = op.get("canonical_description", "")
                if desc:
                    self.insert(desc, int(op.get("value", 1)))

    def snapshot(self) -> dict:
        """JSON-serializable snapshot of the current bank state."""
        return {
            r.rule_id: {"desc": r.canonical_description, "count": r.mention_count}
            for r in self.rules.values()
        }


# ---------------------------------------------------------------------------
# LLM-driven canonicalize-then-match update of R_t
# ---------------------------------------------------------------------------

RULE_EXTRACTION_PROMPT = """You are a rule canonicalization and matching engine (the canonicalize-then-match step that updates RuleBank R_t).

Given a raw textual gradient (feedback on how to improve a system prompt) and the current RuleBank, perform two tasks:

1. CANONICALIZE: Extract mid-level canonical behavioral rules from the raw gradient.
   - Remove references to specific entities, exact numbers, or particular examples.
   - Preserve structural reasoning patterns.
   - Keep rules at mid-level abstraction (not too specific, not too vague).
2. MATCH: For each canonical rule, compare it with the existing RuleBank:
   - If semantically equivalent to an existing rule r (same structural pattern, not just similar wording), output an INCREMENT operation with that rule's ID, which corresponds to m_t(r) = m_{{t-1}}(r) + 1.
   - If no match exists, output an INSERT operation with the canonical description r*, which corresponds to inserting (r*, m_t(r*) = 1).

[CURRENT RULEBANK]
{rulebank_summary}

[RAW GRADIENT]
{raw_gradient}

Output STRICTLY valid JSON matching this schema, nothing else:
{{
    "operations": [
        {{"type": "increment", "rule_id": "R3", "value": 1}},
        {{"type": "insert", "canonical_description": "Always verify intermediate computation steps before producing a final answer", "value": 1}}
    ]
}}"""


def _extract_first_json_object(s: str) -> str:
    """Return the first balanced {...} substring, or '' if none is found."""
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return ""


def update_rulebank_from_gradient(
    rulebank: RuleBank,
    raw_gradient_text: str,
    engine: Any,
) -> None:
    """Canonicalize-then-match one gradient and update R_t accordingly."""
    if not (raw_gradient_text or "").strip():
        return

    prompt = RULE_EXTRACTION_PROMPT.format(
        rulebank_summary=rulebank.get_summary(),
        raw_gradient=raw_gradient_text,
    )
    try:
        reply = engine(prompt)
        reply = str(reply.value).strip() if hasattr(reply, "value") else str(reply).strip()
    except Exception:
        return

    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "operations" in out:
                rulebank.apply_operations(out["operations"])
                return
        except json.JSONDecodeError:
            continue


def update_rulebank_from_gradients(
    rulebank: RuleBank,
    gradients: set,
    engine: Any,
) -> None:
    """Run canonicalize-then-match on every gradient currently attached to a variable."""
    for g in list(gradients):
        update_rulebank_from_gradient(rulebank, getattr(g, "value", "") or "", engine)
