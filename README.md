<div align="center">

# 🪡 TextReg

### *Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization*

[![arXiv](https://img.shields.io/badge/arXiv-2605.21318-b31b1b.svg)](https://arxiv.org/abs/2605.21318)
[![Project Page](https://img.shields.io/badge/Project_Page-textreg.github.io-0e848c.svg?logo=githubpages&logoColor=white)](https://textreg.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-luchengfu6%2FTextReg-181717.svg?logo=github)](https://github.com/luchengfu6/TextReg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![textgrad](https://img.shields.io/badge/textgrad-≥0.1.6-FF6F61.svg)](https://github.com/zou-group/textgrad)
[![OpenAI](https://img.shields.io/badge/OpenAI-API_compatible-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/)

[Lucheng Fu](https://luchengfu6.github.io/)¹, Ye Yu², [Yiyang Wang](https://hello-diana.github.io/)¹, [Yiqiao Jin](https://ahren09.github.io)¹,
[Haibo Jin](https://haibojin001.github.io/)², [B. Aditya Prakash](https://faculty.cc.gatech.edu/~badityap/)¹†, [Haohan Wang](https://haohanwang.github.io/)²†

¹ Georgia Institute of Technology · ² University of Illinois Urbana-Champaign

<sub>† Corresponding authors</sub>

</div>

---

## 📖 Abstract

Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-specific rules, and generalize poorly beyond the training distribution. We study this failure mode as *prompt distributional overfitting* and argue that it reflects a lack of representation control in discrete text-space optimization. We formalize this view through *representational inefficiency*, a dual-factor measure that decomposes prompt inefficiency into capacity cost and scope narrowness, attributing distributional prompt overfitting to their coupled growth during optimization. We propose **TextReg**, a regularization framework that realizes a soft-penalty objective through regularized textual gradients, combining **Dual-Evidence Gradient Purification**, **Semantic Edit Regularization**, and **Regularization-Guided Prompt Update**. Across multiple reasoning benchmarks, **TextReg** substantially improves out-of-distribution (OOD) generalization, with accuracy gains of up to **+11.8% over TextGrad** and **+16.5% over REVOLVE**.

## ✨ Highlights

- 🎯 **First formally defined regularization** for LLM-feedback-based prompt optimization, addressing prompt distributional overfitting
- 🧩 **Three complementary components**: Dual-Evidence Gradient Purification, Semantic Edit Regularization, Regularization-Guided Prompt Update
- 📈 **Strong OOD generalization** with up to **+11.8% over TextGrad** and **+16.5% over REVOLVE** on reasoning benchmarks
- 🔧 **Minimal tuning**: a single hyperparameter ($\tau_C$) controls the entire regularization framework

## 🧩 The TextReg Framework

TextReg decomposes each prompt update into a purified task gradient and a regularization gradient, realized in three stages:

| Stage | What it does | Code |
| :--- | :--- | :--- |
| 🛡️ **1. Dual-Evidence Gradient Purification** | Filter raw task gradients via local batch evidence and global RuleBank recurrence. Classify each as `GENERALIZED_RULE` / `CASE_PATCH` / `STYLE_ONLY`; keep and rewrite generalizable, drop the rest. | [`textreg/purification.py`](textreg/purification.py) |
| 📐 **2. Semantic Edit Regularization** | Estimate per-channel finite differences of representational inefficiency. The active channels (capacity / scope) select a regularization mode and synthesize a regularization gradient. | [`textreg/pipeline.py`](textreg/pipeline.py) |
| 🎯 **3. Regularization-Guided Prompt Update** | Inject the regularization gradient into a `<REG_FEEDBACK>` block; the optimizer LLM picks the task-faithful edit most compatible with it, with task-dominance fallback. | [`textreg/optimizer.py`](textreg/optimizer.py) |

> 📖 For the full formalism (the dual-factor inefficiency measure $\mathcal{I}(p) = C(p)W(p)$, the four active-channel modes, and the projection operator $\Pi_{\text{gen}}$), see the [paper](https://arxiv.org/abs/2605.21318).

## 🚀 Installation

TextReg targets **Python 3.9+** and depends on [textgrad](https://pypi.org/project/textgrad/).

```bash
# 1) Create and activate the env (Python 3.9+ works; 3.11 recommended)
conda create -n textreg python=3.11 -y
conda activate textreg

# 2) Install textreg (also pulls in textgrad, openai, tiktoken, tqdm, ...)
pip install -e .
```

> Full dependency list: [`requirements.txt`](requirements.txt) / [`pyproject.toml`](pyproject.toml). All LLM calls go through `textgrad` engines (OpenAI / Anthropic / local OpenAI-compatible servers).

### Set up API keys

```bash
cp .env.example .env
# then edit .env:
#   OPENAI_API_KEY=sk-...
#   (optional) OPENAI_API_BASE=...   # for Azure / proxies / local OpenAI-compatible servers
#   (optional) HUGGINGFACE_HUB_TOKEN=...
#   (optional) OLLAMA_BASE_URL=...
```

`examples/run_textreg.py` loads `.env` via `python-dotenv` at startup, so you don't need to `export` anything in your shell.

## ⚡ Quick Start

### Option 1 — Reference script

```bash
# Template
python examples/run_textreg.py \
    --task <TASK> \
    --backbone_engine <BACKBONE> \
    --model <SOLVER_MODEL> \
    --batch_size <B> --max_epochs <E> --max_steps <S> \
    --tau_C <TAU_C> \
    --result_dir <OUT_DIR>

# Reproduce the paper's setup:
#   forward engine  = Qwen2.5-7B-Instruct (the model being prompt-optimized)
#   backward engine = GPT-4o (drives all LLM-driven optimization ops)
python examples/run_textreg.py \
    --task BBH_logical_deduction_three_objects \
    --backbone_engine gpt-4o \
    --model ollama-Qwen/Qwen2.5-7B-Instruct \
    --batch_size 3 --max_epochs 1 --max_steps 12 \
    --tau_C 0.2 \
    --result_dir results/
```

For every step the script logs the paper-aligned metrics
(`rho_C`, `sgn_delta_W`, `active_channels`, `reg_mode`, `capacity_C`, `rulebank_size`, `guidance`)
and writes a single JSON file under `--result_dir` with every prompt snapshot, test accuracy, and RuleBank snapshot.

> 💡 Run `python examples/run_textreg.py --help` for the full flag list.

### Option 2 — Use as a library

The four public symbols give you full per-step control inside your own training loop:

```python
import textgrad as tg
from textreg import RuleBank, TextRegOptimizer, apply_textreg_pipeline

backbone = tg.get_engine("gpt-4o")                                # backward + Π_gen + M_Δ + Γ
solver   = tg.get_engine("ollama-Qwen/Qwen2.5-7B-Instruct")       # forward: model being prompt-optimized
tg.set_backward_engine(backbone, override=True)

system_prompt = tg.Variable(
    "You will answer a reasoning question. Think step by step. ...",
    requires_grad=True,
    role_description="system prompt",
)
model     = tg.BlackboxLLM(solver, system_prompt)
optimizer = TextRegOptimizer(engine=backbone, parameters=[system_prompt])
rulebank  = RuleBank()

previous_prompt = system_prompt.value
for batch_x, batch_y in train_loader:
    current_prompt = system_prompt.value
    optimizer.zero_grad()

    losses = [eval_fn(model(x), y) for x, y in zip(batch_x, batch_y)]
    tg.sum(losses).backward()

    metrics = apply_textreg_pipeline(            # Stage 1 + Stage 2
        system_prompt,
        current_prompt=current_prompt,
        previous_prompt=previous_prompt,
        initial_prompt=initial_prompt,
        tokenizer_fn=tokenizer_fn,
        engine=backbone,
        rulebank=rulebank,
        tau_C=0.2,
    )
    optimizer.step()                             # Stage 3
    previous_prompt = current_prompt
```

## 📊 Datasets

The paper evaluates TextReg on **9 reasoning benchmarks** (6 from Big Bench Hard + 3 arithmetic) spanning symbolic deduction and arithmetic. Source tasks are optimized on the easy 3-object / GSM8K variant and evaluated **out-of-distribution** on harder / related variants.

| Dataset | Role | Task family | textgrad identifier |
| :--- | :--- | :--- | :--- |
| **BBH — Logical Deduction (3 obj)** | source (train + val + test) | Symbolic deduction | `BBH_logical_deduction_three_objects` |
| **BBH — Logical Deduction (5 / 7 obj)** | OOD eval (harder variants) | Symbolic deduction | `BBH_logical_deduction_{five,seven}_objects` |
| **BBH — Tracking Shuffled Objects (3 obj)** | source | Symbolic deduction | `BBH_tracking_shuffled_objects_three_objects` |
| **BBH — Tracking Shuffled Objects (5 / 7 obj)** | OOD eval | Symbolic deduction | `BBH_tracking_shuffled_objects_{five,seven}_objects` |
| **GSM8K** | source | Grade-school math | `GSM8K_DSPy` |
| **SVAMP** | OOD eval (robustness variant) | Math word problems | (eval-only) |
| **MultiArith** | OOD eval (multi-step arithmetic) | Math word problems | (eval-only) |

The 3-object BBH variants follow TextGrad's 50 / 100 / 100 train / val / test split.

## 🔧 Hyperparameters

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| `--batch_size` | `3` | Matches REVOLVE protocol. |
| `--max_steps` | `12` | 36 training samples per task total. |
| `--tau_C` | `0.2` | Relative-length-growth threshold for the capacity channel; the **only** TextReg-specific hyperparameter. |

> 📝 **Paper setup**: backward engine = `gpt-4o` (shared with baselines for fair comparison); forward engine = `Qwen2.5-7B-Instruct`; test engines = Qwen2-7B / Phi-3.5-Mini / Llama-3-8B / Llama-3.1-8B (all four evaluated for model-agnostic robustness). TextReg itself does not require a GPU.

## 🧪 Evaluation

`examples/run_textreg.py` performs in-domain test-set evaluation after every step and writes a single JSON file with the full training trajectory:

```json
{
  "final_test_acc_mean":   0.XXXX,
  "rulebank_final":        { "R1": {"desc": "...", "count": N}, ... },
  "pipeline_metrics":      [ { "rho_C": ..., "reg_mode": "...", ... }, ... ],
  "prompt":                [ "<P_0>", "<P_1>", ..., "<P_T>" ]
}
```

For **cross-domain (OOD) evaluation**, load `prompt[-1]` from the result JSON and run it against your held-out dataset via the standard textgrad inference path. The paper evaluates each source-trained prompt across all four test engines listed above to measure model-agnostic robustness.

## 📁 Project layout

```
textreg/
    __init__.py        # public exports
    pipeline.py        # Stage 1 + Stage 2 orchestrator; M_Δ; Γ; state machine
    purification.py    # Stage 1: Π_gen (3-tier gatekeeper)
    rulebank.py        # R_t = {(r, m_t(r))} + canonicalize-then-match
    optimizer.py       # Stage 3: TextRegOptimizer with <REG_FEEDBACK> routing
    prompts.py         # optimizer-side prompt templates (system + reg-guided trailing)
examples/
    run_textreg.py     # reference entry point against any textgrad task
```

## 📝 Citation

If you find TextReg useful in your research, please cite:

```bibtex
@misc{fu2026textregmitigatingpromptdistributional,
      title={TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization},
      author={Lucheng Fu and Ye Yu and Yiyang Wang and Yiqiao Jin and Haibo Jin and B. Aditya Prakash and Haohan Wang},
      year={2026},
      eprint={2605.21318},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.21318},
}
```

## 🙏 Acknowledgements

TextReg builds on the following open-source projects:
[textgrad](https://github.com/zou-group/textgrad) ·
[REVOLVE](https://github.com/Peiyance/REVOLVE) ·
[OpenAI Python SDK](https://github.com/openai/openai-python) ·
[tiktoken](https://github.com/openai/tiktoken).

## ⚖️ License

This project is released under the [MIT License](LICENSE).
