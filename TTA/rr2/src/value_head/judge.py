from __future__ import annotations
from typing import Any
import numpy as np
import torch
from src.common import repeat_4gram_rate
from src.search_common import token_count
from src.value_head.model import extract_candidate_feature_bundle, load_probe_checkpoint, state_action_probe_prompt


def flatten_component_weights(components: list[dict[str, Any]]) -> np.ndarray:
    if not components:
        raise ValueError("Probe v4 requires components.")
    ensemble = np.asarray([float(component.get("ensemble_weight", 1.0)) for component in components])
    if np.any(ensemble < 0) or float(ensemble.sum()) <= 0:
        raise ValueError("Invalid component ensemble weights.")
    ensemble = ensemble / ensemble.sum()
    parts = []
    for component, coefficient in zip(components, ensemble, strict=True):
        weight = component["candidate_weight"]
        if isinstance(weight, torch.Tensor):
            weight = weight.detach().float().cpu().numpy()
        scale = max(float(component.get("score_scale", 1.0)), 1e-12)
        parts.append(np.asarray(weight, dtype=np.float32) * float(coefficient) / scale)
    return np.concatenate(parts).astype(np.float32)


class CandidateHiddenJudge:

    def __init__(self, cfg: dict[str, Any], device: torch.device | None=None, tokenizer: Any | None=None, model: Any | None=None) -> None:
        if tokenizer is None or model is None:
            raise RuntimeError('CandidateHiddenJudge requires the already-loaded SFT tokenizer and model.')
        judge_cfg = cfg['judge']
        probe = load_probe_checkpoint(judge_cfg['probe_path'])
        if probe.get('type') != 'candidate_hidden_probe':
            raise ValueError(f"Unexpected candidate hidden probe type: {probe.get('type')}")
        requested_reward = str(cfg.get('experiment', {}).get('reward_attribute', ''))
        checkpoint_reward = str(probe.get('reward_attribute', ''))
        if checkpoint_reward and requested_reward and (checkpoint_reward != requested_reward):
            raise ValueError(f'Value-head reward mismatch: checkpoint={checkpoint_reward!r}, requested={requested_reward!r}')
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or (torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu'))
        checkpoint_format = str(probe.get('input_format', probe.get('template', '')))
        requested_format = str(judge_cfg.get('input_format', checkpoint_format))
        requested_template = str(judge_cfg.get('template', probe.get('template', requested_format)))
        if checkpoint_format != 'state_action' or requested_format != 'state_action' or requested_template != 'state_action':
            raise ValueError(f'The active value head supports only input_format/template=state_action; checkpoint={checkpoint_format!r}, requested={requested_format!r}/{requested_template!r}.')
        self.template = 'state_action'
        self.batch_size = int(judge_cfg.get('batch_size', 4))
        self.feature_runtime_stats: dict[str, Any] = {'requested_batch_size': self.batch_size, 'final_batch_size': self.batch_size}
        self.input_format = 'state_action'
        self.probe_version = int(probe.get('version', 3))
        if self.probe_version >= 4:
            raw_components = probe.get('components')
            if not isinstance(raw_components, list) or not raw_components:
                raise ValueError('Value-head v4 checkpoint requires non-empty components.')
            self.components: list[dict[str, Any]] = []
            for (index, component) in enumerate(raw_components):
                if str(component.get('feature_view', 'action')) != 'action':
                    raise ValueError(f"Online value-head components must use feature_view=action; component {component.get('name', index)!r} does not.")
                self.components.append({'name': str(component.get('name') or f'component_{index}'), 'layer': int(component['layer']), 'pooling': str(component['pooling']), 'ensemble_weight': float(component.get('ensemble_weight', 1.0)), 'score_scale': float(component.get('score_scale', 1.0)), 'candidate_weight': component['candidate_weight']})
            expected_weight = flatten_component_weights(self.components)
            stored_weight = probe.get('candidate_weight')
            self.weight = expected_weight if stored_weight is None else stored_weight.detach().float().cpu().numpy()
            if self.weight.shape != expected_weight.shape or not np.allclose(self.weight, expected_weight, rtol=1e-05, atol=1e-06):
                raise ValueError('Value-head v4 flat candidate_weight does not match its components.')
            self.layer = int(self.components[0]['layer'])
            self.pooling = str(self.components[0]['pooling'])
        else:
            self.components = []
            self.layer = int(judge_cfg.get('layer', probe['layer']))
            self.pooling = str(judge_cfg.get('pooling', probe['pooling']))
            self.weight = probe['candidate_weight'].detach().float().cpu().numpy()
        hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            component_widths = [int(component['candidate_weight'].numel()) for component in self.components] if self.components else [int(self.weight.shape[0])]
            if any((width != hidden_size for width in component_widths)):
                raise ValueError(f'Value-head feature width does not match the active policy model: checkpoint={component_widths}, model_hidden_size={hidden_size}.')
        self.control_names = list(probe.get('control_names', []))
        control_weight = probe.get('control_weight')
        self.control_weight = control_weight.detach().float().cpu().numpy() if control_weight is not None else None
        self.control_intercept = float(probe.get('control_intercept', 0.0))
        control_mean = probe.get('control_mean')
        control_scale = probe.get('control_scale')
        self.control_mean = control_mean.detach().float().cpu().numpy() if control_mean is not None else None
        self.control_scale = control_scale.detach().float().cpu().numpy() if control_scale is not None else None

    def select(self, prompt: str, prefix: str, candidates: list[str]) -> tuple[int, int]:
        if not candidates:
            raise ValueError('No candidates to select from.')
        scores = self.score(prompt, prefix, candidates)
        best = int(max(range(len(scores)), key=lambda idx: float(scores[idx])))
        return (best, 0)

    def score(self, prompt: str, prefix: str, candidates: list[str]) -> list[float]:
        if not candidates:
            raise ValueError('No candidates to score.')
        features = self.feature_matrix(prompt, prefix, candidates)
        scores = features @ self.weight
        if self.control_weight is not None and self.control_names:
            controls = self.candidate_controls(prefix, candidates)
            scores = scores + controls @ self.control_weight + self.control_intercept
        return [float(score) for score in scores]

    def feature_matrix(self, prompt: str, prefix: str, candidates: list[str]) -> np.ndarray:
        if not candidates:
            raise ValueError('No candidates to featurize.')
        texts = self.feature_texts(prompt, prefix, candidates)
        if self.probe_version >= 4:
            specs = [{'name': component['name'], 'layer': component['layer'], 'pooling': component['pooling']} for component in self.components]
            bundle = extract_candidate_feature_bundle(self.cfg, self.tokenizer, self.model, self.device, texts, specs=specs, batch_size=self.batch_size, runtime_stats=self.feature_runtime_stats)
            return np.concatenate([bundle[component['name']] for component in self.components], axis=1)
        name = f'legacy_{self.layer}_{self.pooling}'
        bundle = extract_candidate_feature_bundle(self.cfg, self.tokenizer, self.model, self.device, texts, specs=[{'name': name, 'layer': self.layer, 'pooling': self.pooling}], batch_size=self.batch_size, runtime_stats=self.feature_runtime_stats)
        return bundle[name]

    def feature_texts(self, prompt: str, prefix: str, candidates: list[str]) -> list[str]:
        return [state_action_probe_prompt(self.tokenizer, prompt, prefix, candidate) for candidate in candidates]

    def candidate_controls(self, prefix: str, candidates: list[str]) -> Any:
        rows = []
        for candidate in candidates:
            values = []
            for name in self.control_names:
                if name == 'candidate_tokens':
                    values.append(float(token_count(self.tokenizer, candidate)))
                elif name == 'prefix_tokens':
                    values.append(float(token_count(self.tokenizer, prefix)))
                elif name == 'candidate_repeat_4gram':
                    values.append(float(repeat_4gram_rate(candidate)))
                elif name == 'empty_prefix':
                    values.append(0.0 if prefix.strip() else 1.0)
                else:
                    raise ValueError(f'Unknown candidate hidden control: {name}')
            rows.append(values)
        controls = torch.tensor(rows, dtype=torch.float32).numpy()
        if self.control_mean is not None and self.control_scale is not None:
            controls = (controls - self.control_mean) / self.control_scale
        return controls

    def close(self) -> None:
        return None
