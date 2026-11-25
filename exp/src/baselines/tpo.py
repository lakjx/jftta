from __future__ import annotations
import re
import time
from typing import Any
import torch
from src.models import generate_texts, render_chat
from src.search_common import generation_row, token_count

TPO_METHODS = {"tpo", "tpo_d2_n3", "tpo_d2_n5"}
TPO_LOSS_SYSTEM_TEMPLATE = "You are a language model tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.\n\n**User Query**:\n{query}\n\n**Rejected Response**:\n{rejected_response}\n\n**Do NOT generate a response to the query. Be concise.** Below is the chosen response."
TEXTGRAD_BACKWARD_SYSTEM_PROMPT = "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). If a variable is already working well, you should not give feedback."
TEXTGRAD_OPTIMIZER_SYSTEM_PROMPT = "You are part of an optimization system that improves text (i.e., variable). You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. You will receive some feedback, and use the feedback to improve the variable. The feedback may be noisy; identify what is important and correct. Pay attention to the role description of the variable and the context in which it is used. You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags. The text between the tags will directly replace the variable."


def build_tpo_loss_system_prompt(query: str, rejected_response: str) -> str:
    return TPO_LOSS_SYSTEM_TEMPLATE.format(
        query=query.strip(), rejected_response=rejected_response.strip()
    )


def build_tpo_backward_prompt(
    loss_system_prompt: str, chosen_response: str, textual_loss: str
) -> str:
    conversation = f"<LM_SYSTEM_PROMPT> {loss_system_prompt} </LM_SYSTEM_PROMPT>\n\n<LM_INPUT> {chosen_response.strip()} </LM_INPUT>\n\n<LM_OUTPUT> {textual_loss.strip()} </LM_OUTPUT>\n\n"
    return f"You will give feedback to a variable with the following role: <ROLE> a chosen response to a user query </ROLE>. Here is an evaluation of the variable using a language model:\n\n{conversation}<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\nWe are interested in giving feedback to the a chosen response to a user query for this conversation. Specifically, give feedback to the following span of text:\n\n<VARIABLE> {chosen_response.strip()} </VARIABLE>\n\nGiven the above history, describe how the a chosen response to a user query could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent."


def build_tpo_update_prompt(
    loss_system_prompt: str,
    chosen_response: str,
    textual_loss: str,
    textual_gradient: str,
) -> str:
    conversation = f"<LM_SYSTEM_PROMPT> {loss_system_prompt} </LM_SYSTEM_PROMPT>\n\n<LM_INPUT> {chosen_response.strip()} </LM_INPUT>\n\n<LM_OUTPUT> {textual_loss.strip()} </LM_OUTPUT>\n\n"
    gradient_context = f"Here is a conversation:\n\n<CONVERSATION>{conversation}</CONVERSATION>\n\nThis conversation is potentially part of a larger system. The output is used as response from the language model\n\nHere is the feedback we got for a chosen response to a user query in the conversation:\n\n<FEEDBACK>{textual_gradient.strip()}</FEEDBACK>\n\n"
    return f"Here is the role of the variable you will improve: <ROLE>a chosen response to a user query</ROLE>.\n\nThe variable is the text within the following span: <VARIABLE> {chosen_response.strip()} </VARIABLE>\n\nHere is the context and feedback we got for the variable:\n\n<CONTEXT>{gradient_context}</CONTEXT>\n\nImprove the variable (a chosen response to a user query) using the feedback provided in <FEEDBACK> tags.\nYou must follow the following constraints:\n\n<CONSTRAINTS>Constraint 1: Only generate a chosen response.\nConstraint 2: Do NOT generate a rejected response.</CONSTRAINTS>\n\nSend the improved variable in the following format:\n\n<IMPROVED_VARIABLE>{{the improved variable}}</IMPROVED_VARIABLE>\n\nSend ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."


def extract_improved_variable(text: str) -> str | None:
    match = re.search(
        "<IMPROVED_VARIABLE>\\s*(.*?)\\s*</IMPROVED_VARIABLE>", text, flags=re.DOTALL
    )
    if match is not None:
        value = match.group(1).strip()
        return value or None
    truncated = re.search("<IMPROVED_VARIABLE>\\s*(.+)\\Z", text, flags=re.DOTALL)
    if truncated is None:
        return None
    value = truncated.group(1).strip()
    return value or None


def select_tpo_contrastive_indices(scores: list[float]) -> tuple[int, int]:
    if not scores:
        raise ValueError("TPO requires at least one scored response.")
    best = max(range(len(scores)), key=lambda index: (float(scores[index]), -index))
    worst = min(range(len(scores)), key=lambda index: (float(scores[index]), index))
    return (best, worst)


def generate_tpo(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    reward: Any,
    device: torch.device,
    prompt: dict[str, Any],
    method: str = "tpo",
) -> dict[str, Any]:
    started = time.monotonic()
    tpo_cfg = resolve_tpo_config(cfg, method)
    iterations = int(tpo_cfg.get("iterations", 2))
    sample_size = int(tpo_cfg.get("samples", 5))
    response_tokens = int(
        tpo_cfg.get("response_max_tokens", cfg["generation"]["max_new_tokens"])
    )
    feedback_tokens = int(tpo_cfg.get("feedback_max_tokens", 512))
    max_input_tokens = int(tpo_cfg.get("max_input_tokens", 7168))
    temperature = float(tpo_cfg.get("temperature", 0.7))
    top_p = float(tpo_cfg.get("top_p", 0.95))
    top_k = int(tpo_cfg.get("top_k", 0))
    initial_chat = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    initial, initial_calls = sample_tpo_outputs(
        cfg,
        tokenizer,
        model,
        device,
        initial_chat,
        sample_size,
        response_tokens,
        max_input_tokens,
        temperature,
        top_p,
        top_k,
    )
    initial_scores = reward.score(
        prompt["prompt"], "", [item["text"] for item in initial]
    )
    cache = [
        candidate_record(item["text"], score, -1, item, tokenizer)
        for item, score in zip(initial, initial_scores, strict=True)
    ]
    lm_calls = initial_calls
    generated_tokens = sum((int(item["raw_token_count"]) for item in initial))
    sampled_responses = len(initial)
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        best, worst = select_tpo_contrastive_indices([item["score"] for item in cache])
        chosen = cache[best]
        rejected = cache[worst]
        loss_system = build_tpo_loss_system_prompt(
            prompt["prompt"], rejected["response"]
        )
        loss_output = generate_tpo_single(
            cfg,
            tokenizer,
            model,
            device,
            loss_system,
            chosen["response"],
            feedback_tokens,
            max_input_tokens,
            temperature,
            top_p,
            top_k,
        )
        backward_prompt = build_tpo_backward_prompt(
            loss_system, chosen["response"], loss_output["text"]
        )
        gradient_output = generate_tpo_single(
            cfg,
            tokenizer,
            model,
            device,
            TEXTGRAD_BACKWARD_SYSTEM_PROMPT,
            backward_prompt,
            feedback_tokens,
            max_input_tokens,
            temperature,
            top_p,
            top_k,
        )
        update_prompt = build_tpo_update_prompt(
            loss_system,
            chosen["response"],
            loss_output["text"],
            gradient_output["text"],
        )
        update_chat = render_chat(
            tokenizer,
            TEXTGRAD_OPTIMIZER_SYSTEM_PROMPT,
            update_prompt,
            response=None,
            add_generation_prompt=True,
        )
        updates, update_calls = sample_tpo_outputs(
            cfg,
            tokenizer,
            model,
            device,
            update_chat,
            sample_size,
            response_tokens,
            max_input_tokens,
            temperature,
            top_p,
            top_k,
        )
        valid: list[tuple[str, dict[str, Any]]] = []
        invalid_outputs: list[str] = []
        unclosed_tag_updates = 0
        for output in updates:
            extracted = extract_improved_variable(output["text"])
            if extracted is None:
                invalid_outputs.append(output["text"])
            else:
                valid.append((extracted, output))
                if "</IMPROVED_VARIABLE>" not in output["text"]:
                    unclosed_tag_updates += 1
        valid_scores = (
            reward.score(prompt["prompt"], "", [text for text, _ in valid])
            if valid
            else []
        )
        new_records = [
            candidate_record(text, score, iteration, output, tokenizer)
            for (text, output), score in zip(valid, valid_scores, strict=True)
        ]
        cache.extend(new_records)
        lm_calls += 2 + update_calls
        sampled_responses += len(updates)
        generated_tokens += int(loss_output["raw_token_count"]) + int(
            gradient_output["raw_token_count"]
        )
        generated_tokens += sum((int(item["raw_token_count"]) for item in updates))
        history.append(
            {
                "iteration": iteration,
                "chosen_cache_index": best,
                "rejected_cache_index": worst,
                "chosen_reward": chosen["score"],
                "rejected_reward": rejected["score"],
                "textual_loss": loss_output["text"],
                "textual_gradient": gradient_output["text"],
                "valid_updates": [item["response"] for item in new_records],
                "valid_update_rewards": [item["score"] for item in new_records],
                "unclosed_tag_updates": unclosed_tag_updates,
                "invalid_update_outputs": invalid_outputs,
            }
        )
    selected_index, _ = select_tpo_contrastive_indices(
        [item["score"] for item in cache]
    )
    selected = cache[selected_index]
    row = generation_row(
        prompt,
        method,
        selected["response"],
        selected["decoded"],
        candidate_idx=selected_index,
    )
    row.update(
        {
            "selection_reward": selected["score"],
            "tpo_variant": method,
            "tpo_iterations": iterations,
            "tpo_samples": sample_size,
            "tpo_temperature": temperature,
            "tpo_top_p": top_p,
            "tpo_cache_size": len(cache),
            "tpo_sampled_responses": sampled_responses,
            "tpo_history": history,
            "lm_forward_calls": lm_calls,
            "tta_generated_tokens": generated_tokens,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row


def resolve_tpo_config(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    if method not in TPO_METHODS:
        raise ValueError(f"Unknown TPO method: {method}")
    decoding = cfg.get("decoding", {})
    result = dict(decoding.get("tpo", {}))
    if method != "tpo":
        variants = decoding.get("tpo_variants", {})
        if method in variants:
            result.update(variants[method])
        elif method != str(cfg.get("method", {}).get("name", "")) or not result:
            raise KeyError(f"Missing decoding.tpo_variants.{method} configuration")
    return result


def generate_tpo_single(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    chat = render_chat(
        tokenizer, system_prompt, user_prompt, response=None, add_generation_prompt=True
    )
    return generate_texts(
        cfg,
        tokenizer,
        model,
        device,
        [chat],
        1,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )[0]


def sample_tpo_outputs(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt_text: str,
    count: int,
    max_new_tokens: int,
    max_input_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    batch_size = max(
        1, min(count, int(cfg.get("decoding", {}).get("candidate_batch_size", count)))
    )
    outputs: list[dict[str, Any]] = []
    calls = 0
    while len(outputs) < count:
        current = min(batch_size, count - len(outputs))
        try:
            outputs.extend(
                generate_texts(
                    cfg,
                    tokenizer,
                    model,
                    device,
                    [prompt_text],
                    current,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            )
            calls += 1
        except torch.cuda.OutOfMemoryError:
            if bool(cfg.get("runtime", {}).get("freeze_batches", False)):
                raise
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
    return (outputs, calls)


def candidate_record(
    response: str,
    score: float,
    iteration: int,
    raw_output: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    decoded = dict(raw_output)
    decoded["text"] = response
    decoded["token_count"] = token_count(tokenizer, response)
    return {
        "response": response,
        "score": float(score),
        "iteration": iteration,
        "decoded": decoded,
    }
