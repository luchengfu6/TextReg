"""
Optimizer prompts for TextReg's Stage 3 (Regularization-Guided Prompt Update).

These prompts drive the LLM that rewrites the system prompt at every
optimizer step. Stage 2 produces a synthetic gradient (g_reg) with a
distinctive role_description; the optimizer surfaces it in a dedicated
<REG_FEEDBACK> block so the LLM can coordinate the purified task gradient
(\\tilde{g}_task) with g_reg. The regularization-guided trailing then asks
the LLM to select, among task-faithful candidates E(p_t, \\tilde{g}_task),
the edit most compatible with g_reg, with a task-dominance fallback when
the two signals are incompatible.
"""

from textgrad.optimizer.optimizer_prompts import (
    GLOSSARY_TEXT,
    OPTIMIZER_SYSTEM_PROMPT,
    TGD_PROMPT_SUFFIX,
    TGD_MULTIPART_PROMPT_INIT,
    TGD_MULTIPART_PROMPT_PREFIX,
    CONSTRAINT_PROMPT_ADDITION,
    IN_CONTEXT_EXAMPLE_PROMPT_ADDITION,
)


# Glossary entry for the new <REG_FEEDBACK> tag, appended to the upstream system prompt.
REG_GLOSSARY_ADDITION = (
    "\n# - <REG_FEEDBACK>: Semantic edit regularization feedback; "
    "follow its structural directives to control prompt length and specificity."
)

OPTIMIZER_SYSTEM_PROMPT_REG = OPTIMIZER_SYSTEM_PROMPT.replace(
    GLOSSARY_TEXT.rstrip(),
    GLOSSARY_TEXT.rstrip() + REG_GLOSSARY_ADDITION,
)


# User-message skeleton. {reg_section} is filled with REG_FEEDBACK_SECTION when
# regularization feedback is non-empty, otherwise with an empty string.
TGD_PROMPT_PREFIX_REG_BASE = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "{reg_section}"
    "Here is the context and task feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
)


# Regularization-guided trailing. Implements Stage 3 of TextReg: the
# optimizer is asked to select an edit from the task-faithful set
# E(p_t, \tilde{g}_task) that is most compatible with g_reg. When task
# feedback and regularization feedback coexist, merges at the same location
# are preferred; at different locations both are applied; under conflict the
# task fix is folded into the reg-aware shape. A specific task item that
# genuinely cannot be addressed within the reg constraints is given
# precedence over that one reg item only (task-dominance fallback); all
# other reg items still apply.
TGD_TRAILING_REG_GUIDED = (
    "Improve the variable ({variable_desc}) by integrating both the task "
    "feedback in <FEEDBACK> and the regularization feedback in "
    "<REG_FEEDBACK>. Both contain concrete instructions; read them together "
    "and apply them as follows:\n"
    "- When the task feedback and the regularization feedback point to the "
    "same part of the prompt, apply them as one combined edit.\n"
    "- When they target different parts, apply both.\n"
    "- When they conflict (e.g., the task feedback asks you to add a new "
    "rule while the regularization feedback asks you to merge or shorten), "
    "address the task feedback in the form most consistent with what the "
    "regularization feedback specifies — for instance, by folding the task "
    "fix into the change that the regularization feedback proposes, so the "
    "two are realized through a single coordinated edit instead of two "
    "unrelated ones.\n"
    "If a specific task feedback item genuinely cannot be addressed without "
    "violating a regularization feedback item, prioritize that task "
    "feedback item and apply it completely, even if this means setting "
    "aside that specific regularization item. All other regularization "
    "items in <REG_FEEDBACK> still apply normally to the rest of the "
    "prompt.\n"
)


# Trailing used when no regularization feedback was injected this step.
TGD_TRAILING_NO_REG = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)


# Reg-feedback section, inserted into the user message when reg feedback is non-empty.
REG_FEEDBACK_SECTION = (
    "The following is semantic edit regularization feedback. "
    "Follow its structural directives to control prompt length and specificity:\n\n"
    "<REG_FEEDBACK>{reg_feedback}</REG_FEEDBACK>\n\n"
)


def construct_tgd_prompt_reg(
    do_constrained: bool = False,
    do_in_context_examples: bool = False,
    reg_feedback: str = "",
    **optimizer_kwargs,
):
    """
    Build the optimizer user message with reg / task feedback separated.

    The variable_grad value may be a string (single-modality) or a list
    (multimodal: list of strings / image bytes). Both paths are handled.
    """
    has_reg = bool(reg_feedback and reg_feedback.strip())
    reg_section = REG_FEEDBACK_SECTION.format(reg_feedback=reg_feedback) if has_reg else ""
    optimizer_kwargs["reg_section"] = reg_section

    trailing = TGD_TRAILING_REG_GUIDED if has_reg else TGD_TRAILING_NO_REG
    prefix_template = TGD_PROMPT_PREFIX_REG_BASE + trailing

    variable_grad = optimizer_kwargs.get("variable_grad")

    if isinstance(variable_grad, str):
        multipart = False
        prompt = prefix_template.format(**optimizer_kwargs)
    else:
        gradient_context = list(variable_grad)
        init_parts = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)]
        if reg_section:
            init_parts.append(reg_section)
        gradient_context = init_parts + gradient_context
        multipart = True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)
        prompt += trailing

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)
    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)
    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt
    return gradient_context + [prompt]
