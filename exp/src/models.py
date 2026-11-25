from __future__ import annotations
import re
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.common import model_reference

def policy_model_path(cfg: dict[str, Any]) -> Any:
    models = cfg.get('models', {})
    path = models.get('policy', models.get('llama'))
    if not path:
        raise KeyError('A policy model is required at models.policy (or legacy models.llama).')
    return path

def load_tokenizer(path: str | Any, *, enable_thinking: bool=False, revision: str | None=None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_reference(path), revision=revision, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        turn_end = chat_turn_end_token(tokenizer)
        tokenizer.pad_token = turn_end if turn_end is not None else tokenizer.eos_token
    tokenizer._tta_enable_thinking = bool(enable_thinking)
    return tokenizer

def load_policy_tokenizer(cfg: dict[str, Any]) -> Any:
    return load_tokenizer(policy_model_path(cfg), enable_thinking=bool(cfg.get('models', {}).get('enable_thinking', False)), revision=cfg.get('models', {}).get('revision'))

def load_lm(cfg: dict[str, Any], device: torch.device, model_path: str | Any | None=None) -> tuple[Any, Any]:
    source_path = model_path or policy_model_path(cfg)
    tokenizer = load_tokenizer(source_path, enable_thinking=bool(cfg.get('models', {}).get('enable_thinking', False)), revision=cfg.get('models', {}).get('revision'))
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        turn_end = chat_turn_end_token(tokenizer)
        tokenizer.pad_token = turn_end if turn_end is not None else tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model_kwargs = {'torch_dtype': dtype, 'low_cpu_mem_usage': True, 'trust_remote_code': True}
    attn_implementation = str(cfg.get('runtime', {}).get('attn_implementation', '')).strip()
    if attn_implementation:
        model_kwargs['attn_implementation'] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_reference(source_path), revision=cfg.get('models', {}).get('revision'), **model_kwargs)
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.max_length = None
    model.eval().to(device)
    return (tokenizer, model)

def render_llama3_chat(system_prompt: str, prompt: str, response: str | None, add_generation_prompt: bool) -> str:
    parts = ['<|begin_of_text|>']
    if system_prompt.strip():
        parts.extend(['<|start_header_id|>system<|end_header_id|>\n\n', system_prompt.strip(), '<|eot_id|>'])
    parts.extend(['<|start_header_id|>user<|end_header_id|>\n\n', prompt.strip(), '<|eot_id|>'])
    parts.extend(['<|start_header_id|>assistant<|end_header_id|>\n\n'])
    if response is not None:
        parts.extend([response.strip(), '<|eot_id|>'])
    elif not add_generation_prompt:
        parts.append('<|eot_id|>')
    return ''.join(parts)

def render_llama3_chat_with_assistant_prefix(system_prompt: str, prompt: str, response_prefix: str) -> str:
    text = render_llama3_chat(system_prompt, prompt, response=None, add_generation_prompt=True)
    return text + response_prefix

def tokenizer_family(tokenizer: Any) -> str:
    if not callable(getattr(tokenizer, 'convert_tokens_to_ids', None)):
        return 'legacy'
    if tokenizer_has_token(tokenizer, '<|eot_id|>'):
        return 'llama3'
    if tokenizer_has_token(tokenizer, '<|im_start|>') and tokenizer_has_token(tokenizer, '<|im_end|>'):
        return 'qwen'
    return 'generic'

def render_chat(tokenizer: Any, system_prompt: str, prompt: str, response: str | None, add_generation_prompt: bool) -> str:
    if tokenizer_family(tokenizer) in {'llama3', 'legacy'}:
        return render_llama3_chat(system_prompt, prompt, response, add_generation_prompt)
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'Tokenizer {type(tokenizer).__name__} has no supported chat template; configure a chat-capable policy tokenizer.')
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({'role': 'system', 'content': system_prompt.strip()})
    messages.append({'role': 'user', 'content': prompt.strip()})
    if response is not None:
        messages.append({'role': 'assistant', 'content': response.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=bool(add_generation_prompt), enable_thinking=bool(getattr(tokenizer, '_tta_enable_thinking', False)))

def render_chat_with_assistant_prefix(tokenizer: Any, system_prompt: str, prompt: str, response_prefix: str) -> str:
    return render_chat(tokenizer, system_prompt, prompt, response=None, add_generation_prompt=True) + response_prefix

def valid_token_id(token_id: Any) -> bool:
    return isinstance(token_id, int) and token_id >= 0

def tokenizer_has_token(tokenizer: Any, token: str) -> bool:
    convert = getattr(tokenizer, 'convert_tokens_to_ids', None)
    convert_back = getattr(tokenizer, 'convert_ids_to_tokens', None)
    if callable(convert) and callable(convert_back):
        try:
            token_id = convert(token)
            return valid_token_id(token_id) and convert_back(token_id) == token
        except (AttributeError, TypeError, ValueError):
            pass
    get_vocab = getattr(tokenizer, 'get_vocab', None)
    if callable(get_vocab):
        try:
            return token in get_vocab()
        except (AttributeError, TypeError, ValueError):
            pass
    if not callable(convert):
        return False
    token_id = convert(token)
    if not valid_token_id(token_id):
        return False
    return not callable(convert_back) or convert_back(token_id) == token

def chat_turn_end_token(tokenizer: Any) -> str | None:
    for token in ('<|eot_id|>', '<|im_end|>'):
        if tokenizer_has_token(tokenizer, token):
            return token
    return tokenizer.eos_token

def chat_turn_end_token_id(tokenizer: Any) -> int:
    token = chat_turn_end_token(tokenizer)
    token_id = tokenizer.convert_tokens_to_ids(token) if token is not None else tokenizer.eos_token_id
    if not valid_token_id(token_id):
        raise ValueError(f'Tokenizer {type(tokenizer).__name__} has no usable turn-end token.')
    return int(token_id)

def generate_texts(cfg: dict[str, Any], tokenizer: Any, model: Any, device: torch.device, prompts: list[str], num_return_sequences: int, max_new_tokens: int | None=None, *, max_input_tokens: int | None=None, temperature: float | None=None, top_p: float | None=None, top_k: int | None=None) -> list[dict[str, Any]]:
    batch_size = int(cfg.get('runtime', {}).get('generation_batch_size', 0))
    if batch_size > 0 and len(prompts) > batch_size:
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(prompts), batch_size):
            outputs.extend(generate_texts(cfg, tokenizer, model, device, prompts[start:start + batch_size], num_return_sequences, max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens, temperature=temperature, top_p=top_p, top_k=top_k))
        return outputs
    encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=int(max_input_tokens or cfg['generation']['max_input_tokens']), add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    input_width = inputs['input_ids'].shape[1]
    stop_ids = generation_stop_token_ids(tokenizer)
    new_tokens = int(max_new_tokens if max_new_tokens is not None else cfg['generation']['max_new_tokens'])
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=True, temperature=float(temperature if temperature is not None else cfg['generation']['temperature']), top_p=float(top_p if top_p is not None else cfg['generation']['top_p']), top_k=int(top_k if top_k is not None else cfg['generation']['top_k']), num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
    return [decode_generated_tokens(cfg, tokenizer, row[input_width:], stop_ids, new_tokens) for row in out]

def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    special_ids = [tokenizer.convert_tokens_to_ids(token) for token in ('<|eot_id|>', '<|im_end|>', '<|end_of_text|>', '<|endoftext|>') if tokenizer_has_token(tokenizer, token)]
    for token_id in (*special_ids, tokenizer.eos_token_id):
        if valid_token_id(token_id) and token_id not in ids:
            ids.append(token_id)
    if not ids and tokenizer.pad_token_id is not None:
        ids.append(int(tokenizer.pad_token_id))
    return ids

def decode_generated_tokens(cfg: dict[str, Any], tokenizer: Any, gen_ids: torch.Tensor, stop_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    ids = [int(token_id) for token_id in gen_ids.detach().cpu().tolist()]
    stop_set = set(stop_ids)
    stop_pos = next((idx for idx, token_id in enumerate(ids) if token_id in stop_set), None)
    effective = ids if stop_pos is None else ids[:stop_pos]
    stop_reason = 'stop_token' if stop_pos is not None else 'max_new_tokens' if len(ids) >= max_new_tokens else 'unknown'
    text = tokenizer.decode(effective, skip_special_tokens=True).strip()
    hit_max = stop_pos is None and len(ids) >= max_new_tokens
    sentence_trimmed = False
    if hit_max and bool(cfg['generation'].get('trim_incomplete_on_max', False)):
        min_tokens = max(1, int(max_new_tokens * float(cfg['generation'].get('trim_min_token_fraction', 0.35))))
        trimmed = trim_to_sentence_boundary(text)
        if trimmed and trimmed != text:
            trimmed_count = len(tokenizer(trimmed, add_special_tokens=False)['input_ids'])
            if trimmed_count >= min_tokens:
                text = trimmed
                effective = tokenizer(text, add_special_tokens=False)['input_ids']
                sentence_trimmed = True
                stop_reason = 'max_new_tokens_sentence_trimmed'
    return {'text': text, 'token_count': len(effective), 'raw_token_count': len(ids) if stop_pos is None else stop_pos, 'hit_max_new_tokens': hit_max, 'sentence_trimmed': sentence_trimmed, 'stop_reason': stop_reason}

def trim_to_sentence_boundary(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    matches = list(re.finditer('[.!?][\\"\')\\]]?(?=\\s|$)', stripped))
    if not matches:
        return stripped
    cut = matches[-1].end()
    if cut < max(32, int(len(stripped) * 0.35)):
        return stripped
    return stripped[:cut].strip()
