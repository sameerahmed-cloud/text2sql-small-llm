"""
Make a model answer exam questions.

The model's answers are saved to a file, then graded separately by evaluate.py.
Keeping the two apart means you can grade on a free CPU runtime and only use the
GPU for the part that needs it (writing the answers).

Colab use (GPU runtime):
    from src.generate import load_model, generate_answers
    llm = load_model("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    preds = generate_answers(llm, records, paths, out_file=".../preds.json")
"""

import json
import time
from pathlib import Path

MAX_MODEL_LEN = 8192   # longest exam/quiz prompt is 7,518 tokens (step 1.3) + room for the answer
MAX_NEW_TOKENS = 256   # the longest answer the model may write; SQL rarely needs more


def load_model(model_id, max_model_len=MAX_MODEL_LEN):
    """Load a model into vLLM, the program that runs it on the GPU.

    dtype="half" = float16. The T4 can't do bfloat16, the default for many models.
    seed=42 so a re-run gives the same answers.
    """
    from vllm import LLM
    llm = LLM(model=model_id, dtype="half", max_model_len=max_model_len, seed=42)
    llm.t2s_max_len = max_model_len  # remembered so generate_answers can check prompt lengths
    return llm


def generate_answers(llm, records, paths, out_file=None, max_new_tokens=MAX_NEW_TOKENS, examples=None):
    """Build each question's prompt, let the model answer all of them, return {id: answer text}.

    temperature=0 means the model always picks its most likely next token
    ("greedy"), so the same prompt always gives the same answer.
    examples: few-shot examples from prompts.load_fewshot_examples (None = zero-shot).
    """
    from vllm import SamplingParams
    from src.data_prep import db_path
    from src.prompts import build_prompt

    tok = llm.get_tokenizer()  # the same tokenizer as in step 1.3
    prompts = [build_prompt(db_path(r, paths), r["question"], r["evidence"], tok, examples)
               for r in records]

    # Check lengths BEFORE generating: one prompt that is too long would stop the whole run
    longest = max(len(tok(p)["input_ids"]) for p in prompts)
    limit = getattr(llm, "t2s_max_len", MAX_MODEL_LEN)
    print(f"Longest prompt: {longest} tokens (+{max_new_tokens} for the answer; limit {limit})")
    if longest + max_new_tokens > limit:
        raise ValueError(f"Prompts too long: reload with load_model(..., max_model_len="
                         f"{longest + max_new_tokens}) or more")

    start = time.time()
    # vLLM answers many prompts at the same time; outputs come back in the same order
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new_tokens))
    minutes = (time.time() - start) / 60

    preds = {r["id"]: o.outputs[0].text for r, o in zip(records, outputs)}
    print(f"Answered {len(preds)} questions in {minutes:.1f} min")

    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(preds, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Saved answers to {out_file}")
    return preds