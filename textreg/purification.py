"""
Stage 1: Dual-Evidence Gradient Purification.

Realizes the conditional projection

    \\tilde{g}_task = Pi_gen(g_task; B_t, R_t)

where Pi_gen filters raw task gradients onto generalizable update
directions using two evidence sources:

  - Local case evidence: the current mini-batch B_t. A gradient that is too
    strongly attributable to idiosyncratic batch examples would inject a
    narrow rule and inflate the scope channel W(p).

  - Global recurrence evidence: the RuleBank R_t with mention counts
    m_t(r). Rules with high recurrence are treated as historically
    generalizable and biased toward acceptance.

Each raw gradient is classified as:

    GENERALIZED_RULE  -> rewritten into a concise broadly applicable
                         instruction and retained.
    CASE_PATCH        -> rejected (would inflate W(p) via a narrow rule).
    STYLE_ONLY        -> rejected (would inflate C(p) without improving scope).

Geometrically Pi_gen is a source-level hard projection: it drops gradient
directions that drive either channel of representational inefficiency
I(p) = C(p) W(p) before any rewrite occurs.

If applying the gate would empty all gradients on a variable, the filter
is bypassed for that step (we keep the original set so the optimizer still
has something to act on). Parse / API failures fall back to keeping the
original gradient.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Set


GRADIENT_GATEKEEPER_SYSTEM = """You are the "Gradient Purifier" (the operator Pi_gen in TextReg's Dual-Evidence Gradient Purification). Your job is to decide whether a proposed feedback (gradient) contains genuinely generalizable improvements, and if so, synthesize them into a concise principle. You output either a purified summary or an empty string.

### INPUT DATA:

<CURRENT_SYSTEM_PROMPT>
{current_prompt}
</CURRENT_SYSTEM_PROMPT>

<EXECUTION_CONTEXT> (The specific batch example -- the local case evidence B_t -- that triggered this gradient)
{gradient_context}
</EXECUTION_CONTEXT>

<PROPOSED_GRADIENT>
{gradient_text}
</PROPOSED_GRADIENT>

<RULEBANK_SUMMARY> (The global recurrence evidence R_t: previously accepted generalizable rules and their mention counts m_t(r))
{rulebank_summary}
</RULEBANK_SUMMARY>

### YOUR TASK:

Internally classify the gradient into one of three categories, then act accordingly.

**Category GENERALIZED_RULE -> OUTPUT PURIFIED TEXT**
The feedback identifies a reasoning flaw, missing constraint, or logical gap that applies broadly across many inputs, not just the specific case in <EXECUTION_CONTEXT>.
- HISTORICAL SIGNAL: If the RuleBank contains a similar rule with high mention_count, this fix has been triggered by many different samples across previous steps -> strong evidence it is generalizable. Lean toward accepting.
- Synthesize into a concise, general principle. Strip specific entities, numbers, and scenario details from the execution context. Merge overlapping points. Each sentence in your output must describe a distinct, actionable behavioral rule -- remove any sentence that merely restates or elaborates on another.
- Substantive rules about output format, reasoning scope, verification steps, and counting methods are generalizable -- do NOT reject them.

**Category CASE_PATCH -> OUTPUT EMPTY STRING ""**
The feedback proposes a fix that is specific to the rare or unusual scenario shown in <EXECUTION_CONTEXT>. Adopting it would add a narrow rule that helps very few future inputs and would inflate the scope channel W(p).
- HISTORICAL SIGNAL: If no similar rule exists in the RuleBank (or mention_count is very low), and the fix is clearly tailored to the specific case in <EXECUTION_CONTEXT>, it is likely a case patch. Lean toward rejecting.
- Examples: "When items are listed in reverse order, count backwards", "If the answer involves a fraction, round down" (when only this one case had fractions).

**Category STYLE_ONLY -> OUTPUT EMPTY STRING ""**
The feedback only concerns tone, formatting, verbosity, or presentation with zero impact on task correctness. Adopting it would inflate the capacity channel C(p) without improving scope.
- Examples: "Use bullet points", "Sound more confident", "Use passive voice".

### OUTPUT FORMAT:
Respond with valid JSON only:
{{
    "purified_gradient": "Your synthesized concise principle (GENERALIZED_RULE), or empty string (CASE_PATCH or STYLE_ONLY)."
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


def _get_context_str(gradients_context: dict, g: Any) -> str:
    """Extract the execution-context string attached to a gradient (if any)."""
    ctx = gradients_context.get(g) if gradients_context is not None else None
    if ctx is None:
        return ""
    c = ctx.get("context")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(part) for part in c)
    return ""


def _call_gatekeeper(
    gradient_text: str,
    gradient_context: str,
    current_prompt: str,
    engine: Any,
    rulebank_summary: str,
    verbose: bool = False,
) -> Optional[dict]:
    """Invoke the LLM realization of Pi_gen and return parsed JSON, or None on failure."""
    prompt = GRADIENT_GATEKEEPER_SYSTEM.format(
        current_prompt=current_prompt or "",
        gradient_context=gradient_context or "",
        gradient_text=gradient_text or "",
        rulebank_summary=rulebank_summary,
    )
    try:
        reply = engine(prompt)
        reply = str(reply.value).strip() if hasattr(reply, "value") else str(reply).strip()
    except Exception:
        return None

    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "purified_gradient" in out:
                return out
        except json.JSONDecodeError:
            continue

    if verbose:
        print("[Purification] Failed to parse gatekeeper reply.")
    return None


def apply_dual_evidence_purification(
    system_prompt: Any,
    engine: Any,
    rulebank: Any = None,
    verbose: bool = False,
) -> dict:
    """
    Realize Pi_gen and apply it in place to system_prompt.gradients.

    For each gradient g_task in the variable's gradient set:
      - GENERALIZED_RULE -> overwrite g_task.value with the purified principle (keep)
      - CASE_PATCH / STYLE_ONLY -> drop
      - Parse / API failure -> keep the original gradient (fallback)

    If applying the filter would empty all gradients, no gradient is
    dropped. The concatenation of the kept gradients realizes
    \\tilde{g}_task in the paper.

    Returns a statistics dict:
        {
            "accepted_texts": list[str],   # purified text for kept gradients
            "n_before": int,
            "n_after": int,
            "n_rejected": int,
            "n_parse_error": int,
        }
    """
    current_prompt = getattr(system_prompt, "value", "") or ""
    gradients_context = getattr(system_prompt, "gradients_context", None) or {}

    rulebank_summary = ""
    if rulebank is not None and hasattr(rulebank, "get_summary"):
        rulebank_summary = rulebank.get_summary()
    if not rulebank_summary:
        rulebank_summary = "No historical data available."

    n_before = len(system_prompt.gradients)
    accepted_texts: List[str] = []
    n_parse_error = 0
    to_remove: Set[Any] = set()

    for g in list(system_prompt.gradients):
        gradient_text = getattr(g, "value", "") or ""
        if not gradient_text.strip():
            to_remove.add(g)
            continue

        result = _call_gatekeeper(
            gradient_text,
            _get_context_str(gradients_context, g),
            current_prompt,
            engine,
            rulebank_summary,
            verbose=verbose,
        )

        if result is None:
            n_parse_error += 1
            if verbose:
                print("[Purification] Parse/API failure -> keep original gradient")
            continue

        purified = (result.get("purified_gradient") or "").strip()
        if purified:
            g.value = purified
            accepted_texts.append(purified)
            if verbose:
                print(f"[Purification] GENERALIZED_RULE ({len(purified)} chars): {purified[:100]}...")
        else:
            if verbose:
                print("[Purification] Rejected (CASE_PATCH or STYLE_ONLY)")
            to_remove.add(g)

    # Safety: if every gradient would be dropped, keep them all to avoid stalling.
    if len(system_prompt.gradients) - len(to_remove) == 0 and to_remove:
        if verbose:
            print("[Purification] Would empty all gradients -> keeping originals.")
        to_remove = set()
    for g in to_remove:
        system_prompt.gradients.discard(g)

    n_after = len(system_prompt.gradients)
    return {
        "accepted_texts": accepted_texts,
        "n_before": n_before,
        "n_after": n_after,
        "n_rejected": n_before - n_after,
        "n_parse_error": n_parse_error,
    }
