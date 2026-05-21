"""
Reference entry point for TextReg prompt optimization on a textgrad task.

Usage:
    python examples/run_textreg.py --task BBH_object_counting --result_dir results/

Environment variables (read by textgrad):
    OPENAI_API_KEY -- required for OpenAI engines.
    Other engine providers (Anthropic, local OpenAI-compatible servers, etc.)
    follow textgrad's standard environment-variable conventions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import textgrad as tg
from dotenv import load_dotenv
from textgrad.tasks import load_task
from tqdm import tqdm

load_dotenv()

# Allow running this script directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from textreg import RuleBank, TextRegOptimizer, apply_textreg_pipeline


# Fallback system prompts for tasks whose textgrad descriptions are empty.
BBH_DEFAULT_PROMPT = (
    "You will answer a reasoning question. Think step by step. The last line "
    "of your response should be of the following format: 'Answer: ($VALUE)' "
    "where VALUE is the letter of the correct option."
)
WORD_SORTING_PROMPT = (
    "You will answer a reasoning question. Think step by step. The last line "
    "of your response must be in the format: 'Answer: [sorted words]', where "
    "[sorted words] are the alphabetically sorted words separated by a single space."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a system prompt with TextReg's adaptive sparse pipeline."
    )
    parser.add_argument(
        "--task", type=str, default="BBH_object_counting",
        help="textgrad task identifier (e.g. BBH_object_counting, GSM8K_DSPy).",
    )
    parser.add_argument(
        "--backbone_engine", type=str, default="gpt-4o",
        help="Engine used for backward feedback, M_Delta, purification, RuleBank, and the optimizer.",
    )
    parser.add_argument(
        "--model", type=str, default="ollama-meta-llama/Llama-3.1-8B-Instruct",
        help="Solver model whose system prompt is being optimized.",
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps", type=int, default=12,
        help="Maximum optimization steps per epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_threads", type=int, default=4,
        help="Worker threads used for parallel sample evaluation.",
    )
    parser.add_argument(
        "--run_validation", action="store_true",
        help="Evaluate the prompt on the validation set after each step.",
    )
    parser.add_argument(
        "--revert_tolerance", type=float, default=0.0,
        help="Revert the step if val accuracy drops by more than this amount.",
    )
    parser.add_argument(
        "--tau_C", type=float, default=0.2,
        help="Capacity threshold tau_C: rho_C > tau_C activates the C channel.",
    )
    parser.add_argument(
        "--result_dir", type=str, default="results",
        help="Directory to save per-step results JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def select_starting_prompt(task: str, train_set) -> str:
    if "word_sorting" in task:
        return WORD_SORTING_PROMPT
    if "object_counting" in task or "GSM8K" in task:
        return train_set.get_task_description()
    return BBH_DEFAULT_PROMPT


def make_engine(engine_name: str, num_threads: int):
    """Construct a textgrad engine, batching where the backend benefits from it."""
    lower_name = (engine_name or "").lower()
    needs_batch = any(k in lower_name for k in ("llama", "qwen", "ollama", "mistral", "gemma"))
    if needs_batch:
        return tg.get_engine(engine_name=engine_name, batch_size=num_threads)
    return tg.get_engine(engine_name=engine_name)


def make_tokenizer():
    """Prefer tiktoken cl100k_base; fall back to whitespace splitting."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s or ""))
    except Exception:
        return lambda s: len((s or "").split())


def eval_sample(item, eval_fn, model):
    x, y = item
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y_var = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x_var)
    try:
        eval_out = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
        return int(eval_out.value)
    except Exception:
        eval_out = eval_fn([x_var, y_var, response])
        return int(eval_fn.parse_output(eval_out))


def eval_dataset(dataset, eval_fn, model, num_threads: int):
    accuracy: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(eval_sample, sample, eval_fn, model) for sample in dataset]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            position=0,
            desc="Evaluating",
        ):
            accuracy.append(future.result())
    return accuracy


def get_eval_output(x, y, model, eval_fn):
    """Forward pass that returns the textgrad eval Variable (needed for .backward())."""
    x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y_var = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x_var)
    try:
        return eval_fn(inputs=dict(prediction=response, ground_truth_answer=y_var))
    except Exception:
        return eval_fn([x_var, y_var, response])


def maybe_revert(system_prompt, results, model, eval_fn, val_set, num_threads, tolerance):
    """Revert the prompt if val accuracy dropped by more than `tolerance`."""
    print("Running the current prompt on the validation set...")
    val_acc = float(np.mean(eval_dataset(val_set, eval_fn, model, num_threads)))
    prev_acc = float(np.mean(results["validation_acc"][-1]))
    print(f"Val acc: {val_acc:.4f}, previous: {prev_acc:.4f}")

    if val_acc < prev_acc - tolerance:
        print(f"Reverting prompt (drop > {tolerance}).")
        system_prompt.set_value(results["prompt"][-1])
        val_acc = prev_acc

    results["validation_acc"].append(val_acc)


def main() -> None:
    args = parse_args()
    print(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    backward_engine = make_engine(args.backbone_engine, args.num_threads)
    solver_engine = make_engine(args.model, args.num_threads)
    tg.set_backward_engine(backward_engine, override=True)

    train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=backward_engine)
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")

    starting_prompt = select_starting_prompt(args.task, train_set)
    print(f"Starting system prompt: {starting_prompt}")

    system_prompt = tg.Variable(
        starting_prompt,
        requires_grad=True,
        role_description=(
            "structured system prompt to a somewhat capable language model that "
            "specifies the behavior and strategies for the QA task"
        ),
    )
    model = tg.BlackboxLLM(solver_engine, system_prompt)

    optimizer = TextRegOptimizer(engine=backward_engine, parameters=[system_prompt])
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    rulebank = RuleBank()
    tokenizer_fn = make_tokenizer()

    results: dict = {
        "args": vars(args),
        "prompt": [],
        "test_acc": [],
        "test_acc_mean": [],
        "validation_acc": [],
        "pipeline_metrics": [],
    }

    print("Evaluating the initial prompt on the test set...")
    results["test_acc"].append(eval_dataset(test_set, eval_fn, model, args.num_threads))
    print("Evaluating the initial prompt on the validation set...")
    results["validation_acc"].append(eval_dataset(val_set, eval_fn, model, args.num_threads))
    results["prompt"].append(system_prompt.get_value())

    previous_prompt = starting_prompt

    for epoch in range(args.max_epochs):
        for step, (batch_x, batch_y) in enumerate(
            (pbar := tqdm(train_loader, position=0))
        ):
            pbar.set_description(f"Epoch {epoch} step {step}")
            current_prompt = system_prompt.value
            optimizer.zero_grad()

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = [
                    executor.submit(get_eval_output, x, y, model, eval_fn)
                    for x, y in zip(batch_x, batch_y)
                ]
                losses = [f.result() for f in concurrent.futures.as_completed(futures)]

            total_loss = tg.sum(losses)
            total_loss.backward()

            metrics = apply_textreg_pipeline(
                system_prompt,
                current_prompt=current_prompt,
                previous_prompt=previous_prompt,
                initial_prompt=starting_prompt,
                tokenizer_fn=tokenizer_fn,
                engine=backward_engine,
                rulebank=rulebank,
                tau_C=args.tau_C,
                verbose=True,
            )
            results["pipeline_metrics"].append(metrics)

            optimizer.step()
            prompt_after_step = system_prompt.get_value()

            if args.run_validation:
                maybe_revert(
                    system_prompt, results, model, eval_fn, val_set,
                    args.num_threads, args.revert_tolerance,
                )

            # Only advance the "previous" pointer when the step was not reverted,
            # otherwise the next M_Delta would compare a prompt to itself.
            if not args.run_validation or system_prompt.get_value() == prompt_after_step:
                previous_prompt = current_prompt

            test_acc = eval_dataset(test_set, eval_fn, model, args.num_threads)
            results["test_acc"].append(test_acc)
            results["test_acc_mean"].append(float(np.mean(test_acc)))
            results["prompt"].append(system_prompt.get_value())

            if step >= args.max_steps - 1:
                break

    results["final_test_acc_mean"] = (
        float(np.mean(results["test_acc"][-1])) if results["test_acc"] else 0.0
    )
    results["rulebank_final"] = rulebank.snapshot()

    print(f"Final test acc : {results['final_test_acc_mean']:.4f}")

    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_short = args.backbone_engine.split("/")[-1]
    model_short = args.model.split("/")[-1]
    filename = f"results_{args.task}_{backbone_short}_{model_short}_{timestamp}.json"
    path = os.path.join(args.result_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
