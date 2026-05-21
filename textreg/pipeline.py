"""
TextReg pipeline.

Realizes the three-stage decomposition of the regularized text-gradient

    g_text(p_t) = \\tilde{g}_task(p_t) + g_reg(p_t),

per the paper:

  Stage 1: Dual-Evidence Gradient Purification
      Filter raw task gradients via local batch evidence and global RuleBank
      recurrence evidence. Realized by `apply_dual_evidence_purification`
      (`textreg.purification`).

  Stage 2: Semantic Edit Regularization (SER)
      Estimate per-channel finite differences of representational
      inefficiency I(p) = C(p) W(p) for the realized transition
      (p_{t-1}, p_t), then synthesize the regularization gradient g_reg as a
      textual instruction.

  Stage 3: Regularization-Guided Prompt Update
      Inject g_reg as a synthetic gradient on the variable and let
      `TextRegOptimizer` route it into <REG_FEEDBACK>; the optimizer LLM then
      selects an edit from the task-faithful set most compatible with g_reg
      (with task-dominance fallback baked into the trailing instruction).

This module implements Stage 2 and exposes `apply_textreg_pipeline`, the
top-level orchestrator that runs Stage 1, Stage 2, and the injection step
that prepares Stage 3.

Capacity channel:
    C(p) is measured as the prompt token count, and triggered by the
    relative length growth rho_C(p_t) = (C(p_t) - C(p_{t-1})) / C(p_{t-1})
    crossing a threshold tau_C.

Scope channel:
    W(p) measures how narrow the rule composition is. The sign of Delta W is
    estimated by the semantic diff analyzer M_Delta (an LLM) that compares
    p_{t-1} and p_t with access to RuleBank and the current gradient
    contexts. The channel triggers when M_Delta reports narrowing
    (sgn(Delta W) = +).

Total LLM calls per step: at most 2 in this module (M_Delta + reg
guidance), on top of the per-gradient gatekeeper / RuleBank calls amortized
by Stage 1.
"""

from __future__ import annotations

import json
from typing import Any, Callable, FrozenSet, Optional

from .purification import apply_dual_evidence_purification
from .rulebank import RuleBank, update_rulebank_from_gradients


# ---------------------------------------------------------------------------
# Synthetic-gradient injection
# ---------------------------------------------------------------------------

# This role_description must start with the marker that
# textreg.optimizer._is_reg_gradient looks for, so the optimizer surfaces
# the directive in the <REG_FEEDBACK> block.
SEMANTIC_REG_ROLE = (
    "Semantic edit regularization directive: follow this instruction when "
    "revising the prompt to control representational inefficiency."
)


def inject_regularization_directive(system_prompt: Any, instruction_text: str) -> None:
    """Attach the regularization gradient g_reg as a synthetic gradient on the variable."""
    import textgrad as tg

    var = tg.Variable(
        value=instruction_text,
        requires_grad=False,
        role_description=SEMANTIC_REG_ROLE,
    )
    system_prompt.gradients.add(var)
    if getattr(system_prompt, "gradients_context", None) is None:
        system_prompt.gradients_context = {}
    system_prompt.gradients_context[var] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _llm_call(engine: Any, prompt: str) -> str:
    """Call an engine and normalize the reply to a string."""
    try:
        reply = engine(prompt)
    except Exception:
        return ""
    return str(reply.value).strip() if hasattr(reply, "value") else str(reply).strip()


def _collect_gradient_contexts(system_prompt: Any, max_contexts: int = 3) -> str:
    """Extract up to N execution-context snippets attached to the gradients (G_t)."""
    gradients_context = getattr(system_prompt, "gradients_context", None) or {}
    contexts: list[str] = []
    for i, g in enumerate(list(system_prompt.gradients)[:max_contexts]):
        ctx = gradients_context.get(g)
        if ctx is None:
            continue
        c = ctx.get("context") if isinstance(ctx, dict) else ctx
        if c is None:
            continue
        ctx_str = c if isinstance(c, str) else str(c)
        contexts.append(f"[Context {i+1}] {ctx_str[:500]}")
    return "\n".join(contexts) if contexts else "No gradient contexts available."


# ---------------------------------------------------------------------------
# M_Delta: semantic diff analyzer (scope channel of Stage 2)
# ---------------------------------------------------------------------------

SEMANTIC_DELTA_SYSTEM = """You are the "Semantic Delta Analyzer". Compare PREVIOUS_PROMPT and CURRENT_PROMPT at the level of behavioral rules, classify each change, and judge the overall specificity shift.

WHAT YOU MUST NOT DO:
1. DO NOT judge whether the task logic or the new rules are correct.
2. DO NOT act as a prompt quality evaluator.
3. DO NOT produce a character-level or word-level diff. Focus on rule-level structural changes.

WHAT YOU MUST DO:
1. Identify each structural/behavioral change between PREVIOUS_PROMPT and CURRENT_PROMPT (rule-level, not wording).
2. Classify each change. IMPORTANT: Default to CASE_PATCH unless there is clear positive evidence for GENERALIZED_RULE.

Classification criteria:

GENERALIZED_RULE -- A broadly applicable, task-agnostic behavioral principle that is likely useful across many inputs. Mark as GENERALIZED_RULE if:
- It is NOT tied to specific entities, exact numbers, named templates, or surface strings, AND
- It is a reusable reasoning/decision primitive (not a rare exception branch), AND
- It has at least ONE strong support signal:
  (a) a semantically similar high-frequency RuleBank entry (high mention_count), OR
  (b) it appears relevant across multiple execution contexts in GRADIENT_CONTEXTS (at least two distinct contexts).

Otherwise, default to CASE_PATCH.

CASE_PATCH -- The DEFAULT for new/modified rules. Typical signs:
- The change is best explained as being triggered by a single specific example in GRADIENT_CONTEXTS, with no clear evidence it would apply beyond that example.
- There is weak historical support: no semantically similar high-frequency rule in RuleBank (low mention_count), and no evidence the rule applies across multiple contexts.
- The change introduces extra constraints/conditions that mainly resolve a localized failure pattern rather than improving broadly reusable behavior.

STYLE_ONLY -- Pure wording/formatting change with no behavioral impact.

3. After classifying all changes, judge the OVERALL specificity direction (the sign of Delta W):
- "increase": The update adds or strengthens patch-like rules overall (scope has narrowed).
- "decrease": The update removes CASE_PATCH rules or replaces them with GENERALIZED_RULEs, AND does NOT add new CASE_PATCH rules (scope has broadened).
- "neutral": No meaningful net change (only STYLE_ONLY changes, or mixed changes that roughly offset).

[INPUTS]
<INITIAL_PROMPT>
{initial_prompt}
</INITIAL_PROMPT>

<PREVIOUS_PROMPT>
{previous_prompt}
</PREVIOUS_PROMPT>

<CURRENT_PROMPT>
{current_prompt}
</CURRENT_PROMPT>

<RULEBANK_SUMMARY>
{rulebank_summary}
</RULEBANK_SUMMARY>

<GRADIENT_CONTEXTS> (Execution contexts from the previous iteration showing what inputs/outputs triggered the changes)
{gradient_contexts}
</GRADIENT_CONTEXTS>

[OUTPUT FORMAT]
Return strictly a JSON object matching this schema. Do not output anything else.
{{
    "rules_changed": [
        {{
            "description": "Brief description of what changed",
            "type": "GENERALIZED_RULE | CASE_PATCH | STYLE_ONLY"
        }}
    ],
    "specificity_direction": "increase | decrease | neutral"
}}"""


def run_M_delta_analyzer(
    previous_prompt: str,
    current_prompt: str,
    initial_prompt: str,
    engine: Any,
    rulebank_summary: str,
    gradient_contexts: str,
) -> Optional[dict]:
    """
    M_Delta(p_{t-1}, p_t, R_t, G_t) -> (rules_changed, sgn(Delta W)).

    Returns a dict with keys "rules_changed" and "specificity_direction", or
    None on parse failure / identical prompts being a no-op.
    """
    if not (previous_prompt or "").strip() or not (current_prompt or "").strip():
        return None
    if previous_prompt.strip() == current_prompt.strip():
        return {"rules_changed": [], "specificity_direction": "neutral"}

    prompt = SEMANTIC_DELTA_SYSTEM.format(
        initial_prompt=(initial_prompt or "")[:4000],
        previous_prompt=(previous_prompt or "")[:4000],
        current_prompt=(current_prompt or "")[:4000],
        rulebank_summary=rulebank_summary or "No historical data available.",
        gradient_contexts=gradient_contexts or "No gradient contexts available.",
    )
    reply = _llm_call(engine, prompt)
    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "rules_changed" in out:
                return out
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Per-channel triggers + active-channel set A_t
# ---------------------------------------------------------------------------

def compute_rho_C(
    current_prompt: str,
    previous_prompt: str,
    tokenizer_fn: Callable[[str], int],
) -> float:
    """Relative length growth rho_C(p_t) = (C(p_t) - C(p_{t-1})) / C(p_{t-1})."""
    n_cur = tokenizer_fn(current_prompt or "")
    if not (previous_prompt and previous_prompt.strip()):
        return 0.0
    n_prev = tokenizer_fn(previous_prompt.strip())
    if n_prev <= 0:
        return 0.0
    return (n_cur - n_prev) / n_prev


def scope_change_sign(M_delta_result: Optional[dict]) -> int:
    """sgn(Delta W) in {+1, 0, -1}, decoded from the M_Delta `specificity_direction`."""
    if not M_delta_result:
        return 0
    direction = M_delta_result.get("specificity_direction", "neutral")
    if isinstance(direction, str):
        direction = direction.strip().lower()
    if direction == "increase":
        return 1
    if direction == "decrease":
        return -1
    return 0


# Mode names (see paper Sec. 4.3 and Appendix D.4). These correspond
# one-to-one with the active-channel set A_t:
#   {C, W} -> STRONG_REGULARIZATION  (both compress AND generalize)
#   {C}    -> COMPRESSION_ONLY       (compress only)
#   {W}    -> GENERALIZE_ONLY        (generalize only)
#   {}     -> NO_REGULARIZATION      (skip the Gamma LLM call)
STRONG_REGULARIZATION = "STRONG_REGULARIZATION"
COMPRESSION_ONLY = "COMPRESSION_ONLY"
GENERALIZE_ONLY = "GENERALIZE_ONLY"
NO_REGULARIZATION = "NO_REGULARIZATION"


def active_channels(rho_C_value: float, sgn_delta_W: int, tau_C: float = 0.2) -> FrozenSet[str]:
    """
    Active-channel set A_t = {C : b_C(p_t)=1} U {W : b_W(p_t)=1}.

    b_C(p_t) = 1 iff rho_C(p_t) > tau_C.
    b_W(p_t) = 1 iff sgn(Delta W) = +1.
    """
    active: set[str] = set()
    if rho_C_value > tau_C:
        active.add("C")
    if sgn_delta_W == 1:
        active.add("W")
    return frozenset(active)


def channels_to_mode(channels: FrozenSet[str]) -> str:
    """Map A_t to one of the four mode names above."""
    if channels == frozenset({"C", "W"}):
        return STRONG_REGULARIZATION
    if channels == frozenset({"C"}):
        return COMPRESSION_ONLY
    if channels == frozenset({"W"}):
        return GENERALIZE_ONLY
    return NO_REGULARIZATION


# ---------------------------------------------------------------------------
# Gamma: regularization gradient generator
# ---------------------------------------------------------------------------

REGULARIZATION_GUIDANCE_PROMPT = """You are a structural regularization controller (the operator Gamma in the SER stage).

You are given a REGULARIZATION_MODE (the active-channel set A_t of representational inefficiency I(p) = C(p) W(p)), the current prompt, and a list of recent rule changes. Generate precise structural regularization guidance strictly according to the specified mode.

<REGULARIZATION_MODE>
{mode}
</REGULARIZATION_MODE>

Mode definitions (follow the one that matches; see paper Sec. 4.3):

STRONG_REGULARIZATION (A_t = {{C, W}}):
  Capacity grew (rho_C > tau_C) AND scope narrowed (sgn(Delta W) = +).
  This is the most critical situation -- the prompt is both bloating and
  becoming more specialized.
  You MUST give firm directives to BOTH compress AND generalize:
  (1) Merge redundant sentences that express the same behavioral rule into a single shorter statement.
  (2) Tighten verbose phrasing -- convey the same meaning in fewer tokens.
  (3) Identify narrow rules that target specific rare scenarios and rewrite them as broader principles that cover a wider range of inputs.
  (4) Remove case-specific patches that cannot be meaningfully generalized. However, do NOT remove rules that are broadly useful -- only compress their expression while preserving their behavioral intent.
  Tone: firm imperative.

COMPRESSION_ONLY (A_t = {{C}}):
  Capacity grew but scope did not narrow. The prompt is bloating and must be shortened.
  You MUST focus ONLY on reducing prompt length:
  (1) Merge redundant sentences that express the same behavioral rule into a single shorter statement.
  (2) Tighten verbose phrasing -- convey the same meaning in fewer tokens.
  Do NOT remove rules that are broadly useful -- compress their expression while preserving their behavioral intent.
  Tone: moderate directive.

GENERALIZE_ONLY (A_t = {{W}}):
  Capacity is stable but scope narrowed. The prompt is becoming more specialized without growing longer.
  You MUST focus ONLY on generalization:
  (1) Identify narrow rules that target specific rare scenarios.
  (2) Rewrite them as broader principles that cover a wider range of inputs.
  Length compression is not the goal -- focus on broadening scope.
  Tone: moderate directive.

<CURRENT_PROMPT>
{current_prompt}
</CURRENT_PROMPT>

<NEWLY_CHANGED_RULES>(Rule changes detected by M_Delta between the previous and current prompt, each tagged as GENERALIZED_RULE, CASE_PATCH, or STYLE_ONLY)
{rules_changed_summary}
</NEWLY_CHANGED_RULES>

Rules for generating guidance:
- Do NOT alter task semantics or remove rules that are clearly beneficial for task correctness.
- You MUST reference specific sentences or rules from <CURRENT_PROMPT> and <NEWLY_CHANGED_RULES> when suggesting merges, removals, or generalizations. Do NOT give generic advice -- say exactly WHICH rules to change and HOW.
- Be concise. Do not repeat or rephrase the same suggestion. Do not include explanations or justifications -- state what to change and how, nothing more.

Output STRICTLY valid JSON:
{{
    "guidance": "Your concise regularization guidance referencing specific rules."
}}"""


def _summarize_rules_changed(M_delta_result: Optional[dict]) -> str:
    if not M_delta_result or not M_delta_result.get("rules_changed"):
        return "No rule changes detected."
    lines: list[str] = []
    for r in M_delta_result["rules_changed"]:
        if isinstance(r, dict):
            desc = r.get("description", "unknown change")
            rtype = r.get("type", "UNKNOWN")
            lines.append(f"- [{rtype}] {desc}")
        elif isinstance(r, str):
            lines.append(f"- {r}")
    return "\n".join(lines) if lines else "No rule changes detected."


def synthesize_g_reg(
    mode: str,
    rules_changed_summary: str,
    current_prompt: str,
    engine: Any,
) -> Optional[str]:
    """
    Gamma(rules_changed, A_t) -> g_reg.

    Ask the LLM-realized generator Gamma to produce structural guidance
    matched to the active-channel mode. Returns None when A_t is empty.
    """
    if mode == NO_REGULARIZATION:
        return None

    prompt = REGULARIZATION_GUIDANCE_PROMPT.format(
        mode=mode,
        current_prompt=(current_prompt or "")[:3000],
        rules_changed_summary=(rules_changed_summary or "No rule changes detected.")[:1500],
    )
    reply = _llm_call(engine, prompt)
    for candidate in [reply, _extract_first_json_object(reply)]:
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            if isinstance(out, dict) and "guidance" in out:
                g = out["guidance"]
                if isinstance(g, list):
                    g = "\n".join(str(item) for item in g if item is not None)
                elif not isinstance(g, str):
                    g = str(g)
                guidance = g.strip()
                return guidance or None
        except json.JSONDecodeError:
            continue

    # JSON parse failed: only fall back to the raw reply when it is short
    # enough to plausibly be the guidance text itself.
    if reply and len(reply) < 500:
        return reply
    return None


# ---------------------------------------------------------------------------
# Main entry point: Stage 1 + Stage 2 + injection for Stage 3
# ---------------------------------------------------------------------------

def apply_textreg_pipeline(
    system_prompt: Any,
    current_prompt: str,
    previous_prompt: str,
    initial_prompt: str,
    tokenizer_fn: Callable[[str], int],
    engine: Any,
    rulebank: Optional[RuleBank] = None,
    tau_C: float = 0.2,
    verbose: bool = False,
) -> dict:
    """
    Run Stage 1 + Stage 2 of TextReg for one optimization step.

    Order of operations per the paper:

      (Stage 1) Dual-Evidence Gradient Purification
                -> filters system_prompt.gradients in place and updates R_t.
      (Stage 2) Semantic Edit Regularization
                a. M_Delta(p_{t-1}, p_t, R_t, G_t) -> (rules_changed, sgn(Delta W))
                b. rho_C(p_t) from tokenizer_fn
                c. Active channels A_t and the corresponding mode name
                d. If A_t != {{}}, Gamma -> g_reg, then inject as synthetic gradient

    Stage 3 (regularization-guided rewrite) is performed by
    `TextRegOptimizer.step()` on the variable whose gradients we just updated.

    Args:
        system_prompt:    textgrad Variable being optimized (carries .gradients).
        current_prompt:   String value of system_prompt at the start of this step (p_t).
        previous_prompt:  String value of system_prompt at the start of the previous step (p_{t-1}).
        initial_prompt:   String value of the very first system_prompt (p_0).
        tokenizer_fn:     Callable that returns a token count -- realizes C(p).
        engine:           textgrad engine used for all LLM calls in this pipeline.
        rulebank:         Optional RuleBank for the global recurrence prior.
        tau_C:            Capacity threshold; rho_C > tau_C activates the C channel.
        verbose:          Print per-step intermediate signals.

    Returns a dict of diagnostic metrics indexed by paper-aligned names:
        rho_C, sgn_delta_W, active_channels, reg_mode, capacity_C,
        rulebank_size, guidance.
    """
    # Snapshot context strings before purification mutates anything.
    gradient_contexts_str = _collect_gradient_contexts(system_prompt)

    # RuleBank summary (or a neutral placeholder when not provided).
    rulebank_summary = ""
    if rulebank is not None:
        rulebank_summary = rulebank.get_summary()
    if not rulebank_summary:
        rulebank_summary = "(empty)" if rulebank is not None else "No historical data available."

    # ---- Stage 2 (a): M_Delta semantic diff analyzer (LLM call #1) ----
    delta_result = run_M_delta_analyzer(
        previous_prompt,
        current_prompt,
        initial_prompt,
        engine,
        rulebank_summary=rulebank_summary,
        gradient_contexts=gradient_contexts_str,
    )
    if verbose:
        if delta_result:
            print("[TextReg] M_Delta rules_changed:", delta_result.get("rules_changed", []))
            print("[TextReg] M_Delta sgn(Delta W):", delta_result.get("specificity_direction", "N/A"))
        else:
            print("[TextReg] M_Delta returned None (parse error or identical prompts)")

    sgn_delta_W = scope_change_sign(delta_result)
    rho_C_value = compute_rho_C(current_prompt, previous_prompt, tokenizer_fn)
    A_t = active_channels(rho_C_value, sgn_delta_W, tau_C=tau_C)
    reg_mode = channels_to_mode(A_t)

    if verbose:
        print(
            f"[TextReg] rho_C={rho_C_value:.4f} (tau_C={tau_C}), "
            f"sgn(Delta W)={sgn_delta_W}, A_t={set(A_t) or '{{}}'}, mode={reg_mode}"
        )

    # ---- Stage 1: Dual-Evidence Gradient Purification + RuleBank update ----
    purification_result = apply_dual_evidence_purification(
        system_prompt, engine, rulebank=rulebank, verbose=verbose,
    )
    if rulebank is not None and purification_result["accepted_texts"]:
        if verbose:
            print(
                f"[TextReg] Canonicalize-then-match: updating RuleBank from "
                f"{len(purification_result['accepted_texts'])} purified gradients..."
            )
        update_rulebank_from_gradients(rulebank, system_prompt.gradients, engine)
        if verbose:
            print(f"[TextReg] |R_t| = {len(rulebank.rules)} rules")

    # ---- Stage 2 (b): Gamma synthesizes g_reg (LLM call #2) ----
    rules_summary = _summarize_rules_changed(delta_result)
    guidance = synthesize_g_reg(reg_mode, rules_summary, current_prompt, engine)
    if verbose:
        print(f"[TextReg] g_reg: {guidance}")

    # ---- Inject g_reg as a synthetic gradient on the variable for Stage 3 ----
    if guidance:
        inject_regularization_directive(system_prompt, guidance)

    return {
        "rho_C": rho_C_value,
        "sgn_delta_W": sgn_delta_W,
        "active_channels": sorted(A_t),
        "reg_mode": reg_mode,
        "capacity_C": tokenizer_fn(current_prompt),
        "rulebank_size": len(rulebank.rules) if rulebank else 0,
        "guidance": guidance,
    }
