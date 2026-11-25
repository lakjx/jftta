from __future__ import annotations

from src.models import generate_texts, render_chat
from src.search_common import generation_row

METHODS = ("sft", "grpo", "cos", "tmpc", "tpo", "genarm", "bon")


def generate(method, cfg, tokenizer, model, judge, device, prompt):
    if method in {"sft", "grpo"}:
        chat = render_chat(
            tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
        )
        output = generate_texts(cfg, tokenizer, model, device, [chat], 1)[0]
        return generation_row(
            prompt, method, output["text"], output, candidate_idx=None
        )
    if method == "bon":
        from .bon import generate_bon

        return generate_bon(cfg, tokenizer, model, judge, device, prompt)
    if method == "cos":
        from .cos import generate_cos

        return generate_cos(cfg, tokenizer, model, device, prompt)
    if method == "tmpc":
        from .tmpc import generate_tmpc_subgoal_regeneration

        return generate_tmpc_subgoal_regeneration(
            cfg, tokenizer, model, judge, device, prompt
        )
    if method == "tpo":
        from .tpo import generate_tpo

        return generate_tpo(cfg, tokenizer, model, judge, device, prompt)
    if method == "genarm":
        from .genarm import generate_genarm

        return generate_genarm(cfg, tokenizer, model, device, prompt)
    raise ValueError(f"Unknown baseline {method!r}; choose from {METHODS}")
