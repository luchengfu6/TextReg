"""
Stage 3: Regularization-Guided Prompt Update.

TextRegOptimizer is a TextualGradientDescent subclass that implements the
final composition step

    g_text(p_t) = \\tilde{g}_task(p_t) + g_reg(p_t)

with task-dominance fallback. It routes the two textual gradients into
separate blocks of the optimizer's user message:

  - \\tilde{g}_task (the purified task gradient produced by Stage 1) is
    rendered in <CONTEXT>, exactly as upstream textgrad does.
  - g_reg (the regularization gradient synthesized by Stage 2 and injected
    onto the variable by the pipeline) is identified by its
    role_description marker (SEMANTIC_REG_ROLE_MARKER) and lifted into a
    dedicated <REG_FEEDBACK> block.

The trailing instruction in `textreg.prompts` then directs the optimizer
LLM to select, among task-faithful candidates E(p_t, \\tilde{g}_task), the
edit most compatible with g_reg -- and to fall back to a task-faithful
edit whenever the two signals are incompatible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Union

from textgrad import logger
from textgrad.config import validate_engine_or_get_default
from textgrad.engine import EngineLM
from textgrad.optimizer.optimizer import Optimizer
from textgrad.optimizer.optimizer_prompts import (
    GRADIENT_MULTIPART_TEMPLATE,
    GRADIENT_TEMPLATE,
)
from textgrad.variable import Variable

from .prompts import OPTIMIZER_SYSTEM_PROMPT_REG, construct_tgd_prompt_reg


# Substring that marks a gradient as regularization-generated. The pipeline
# writes a role_description starting with this marker when it injects a
# synthetic gradient. The optimizer matches case-insensitively.
SEMANTIC_REG_ROLE_MARKER = "Semantic edit regularization directive"


def _is_reg_gradient(g) -> bool:
    role = (getattr(g, "role_description", "") or "").lower()
    return SEMANTIC_REG_ROLE_MARKER.lower() in role


def split_reg_and_task_gradients(variable: Variable) -> Tuple[str, Union[str, list]]:
    """
    Split variable.gradients into the two components of g_text:

      - reg_feedback (g_reg): concatenated text of all regularization
        gradients identified by `_is_reg_gradient`.
      - task_grad (\\tilde{g}_task): purified task gradients rendered with
        their execution contexts (string when all entries are strings,
        otherwise a list that the multipart optimizer prompt expects).
    """
    reg_parts: list[str] = []
    task_grads: list = []
    for g in variable.gradients:
        if _is_reg_gradient(g):
            reg_parts.append((g.value or "").strip())
        else:
            task_grads.append(g)

    reg_feedback = "\n".join(p for p in reg_parts if p).strip()

    rendered: list = []
    context_dict = getattr(variable, "gradients_context", None) or {}
    for g in task_grads:
        ctx = context_dict.get(g)
        if ctx is None:
            rendered.append(g.value)
            continue

        ctx_payload = ctx.get("context")
        if isinstance(ctx_payload, str):
            rendered.append(GRADIENT_TEMPLATE.format(feedback=g.value, **ctx))
        elif isinstance(ctx_payload, list):
            multipart = GRADIENT_MULTIPART_TEMPLATE.format(**ctx, feedback=g.value)
            rendered.extend(ctx_payload + [multipart])
        else:
            raise ValueError("Gradient context must be either a string or a list.")

    task_grad: Union[str, list]
    if all(isinstance(item, str) for item in rendered):
        task_grad = "\n".join(rendered) if rendered else ""
    else:
        task_grad = rendered
    return reg_feedback, task_grad


class TextRegOptimizer(Optimizer):
    """
    Textual gradient descent with regularization-aware routing (Stage 3).

    Differences from upstream TextualGradientDescent:
      1. Gradients marked as g_reg (regularization gradient from Stage 2)
         are rendered in a separate <REG_FEEDBACK> block instead of being
         concatenated into <CONTEXT>.
      2. When g_reg is present, the user message ends with the
         regularization-guided trailing (TGD_TRAILING_REG_GUIDED), which
         instructs the LLM to select an edit from the task-faithful set
         E(p_t, \\tilde{g}_task) that is most compatible with g_reg, and
         to fall back to a task-faithful edit when the two signals are
         incompatible (task-dominance fallback).
    """

    def __init__(
        self,
        parameters: List[Variable],
        engine: Union[EngineLM, str] = None,
        verbose: int = 0,
        constraints: List[str] = None,
        in_context_examples: List[str] = None,
        new_variable_tags: List[str] = None,
        optimizer_system_prompt: str = OPTIMIZER_SYSTEM_PROMPT_REG,
    ):
        super().__init__(parameters)

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints or []
        self.in_context_examples = in_context_examples or []
        self.new_variable_tags = new_variable_tags
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0],
            new_variable_end_tag=new_variable_tags[1],
        )
        self.do_constrained = len(self.constraints) > 0
        self.do_in_context_examples = len(self.in_context_examples) > 0
        # Kept for parity with the upstream Optimizer interface.
        self.gradient_memory_dict: dict = defaultdict(list)

    @property
    def constraint_text(self) -> str:
        return "\n".join(
            f"Constraint {i + 1}: {c}" for i, c in enumerate(self.constraints)
        )

    def _update_prompt(self, variable: Variable) -> Union[str, list]:
        reg_feedback, task_grad = split_reg_and_task_gradients(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": task_grad,
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
        }

        prompt = construct_tgd_prompt_reg(
            do_constrained=self.do_constrained,
            do_in_context_examples=self.do_in_context_examples,
            reg_feedback=reg_feedback,
            **optimizer_information,
        )

        logger.info(
            "TextRegOptimizer prompt for update", extra={"prompt": prompt}
        )
        return prompt

    def step(self) -> None:
        for parameter in self.parameters:
            user_prompt = self._update_prompt(parameter)
            new_text = self.engine(user_prompt, system_prompt=self.optimizer_system_prompt)
            logger.info(
                "TextRegOptimizer optimizer response",
                extra={"optimizer.response": new_text},
            )
            if self.verbose:
                print("--- TextReg optimizer user prompt ---")
                print(user_prompt)
                print("--- TextReg optimizer response ---")
                print(new_text)

            start_tag, end_tag = self.new_variable_tags
            try:
                new_value = new_text.split(start_tag)[1].split(end_tag)[0].strip()
            except IndexError as exc:
                logger.error(
                    "TextRegOptimizer response could not be parsed",
                    extra={"optimizer.response": new_text},
                )
                raise IndexError(
                    "TextRegOptimizer response missing improved-variable tags. "
                    f"Response: {new_text}"
                ) from exc

            parameter.set_value(new_value)
            logger.info(
                "TextRegOptimizer updated text",
                extra={"parameter.value": parameter.value},
            )
