"""
TextReg: adaptive sparse regularization for textual gradient descent.

The package decomposes each prompt update into a purified task gradient
\\tilde{g}_task and a regularization gradient g_reg, realized in three
stages (see README and `textreg.pipeline`).

Public entry points:
    RuleBank                          -- cross-step memory R_t with m_t(r)
    TextRegOptimizer                  -- Stage 3 optimizer (<REG_FEEDBACK> routing)
    apply_textreg_pipeline            -- Stage 1 + Stage 2 (one optimization step)
    apply_dual_evidence_purification  -- Stage 1 only (Pi_gen gatekeeper)
"""

from .optimizer import TextRegOptimizer
from .pipeline import apply_textreg_pipeline
from .purification import apply_dual_evidence_purification
from .rulebank import RuleBank

__all__ = [
    "RuleBank",
    "TextRegOptimizer",
    "apply_textreg_pipeline",
    "apply_dual_evidence_purification",
]
