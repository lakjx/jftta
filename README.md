# OVAL

**Online Value Adaptation for Test-Time Alignment**

OVAL improves inference-time search by turning lookahead continuations into supervision for a lightweight value correction. The language model and initial critic remain frozen. Only a prompt-specific residual is updated during generation, and its accumulated evidence carries forward across response segments.

## Method

For each prompt, OVAL repeatedly:

1. Samples candidate segments conditioned on the prompt and the response prefix.
2. Generates bounded lookahead continuations and scores them with the initial critic.
3. Backs up continuation scores to form pairwise preferences between candidate segments.
4. Accumulates preference gradients and updates a regularized residual using Follow-the-Regularized-Leader (FTRL).
5. Selects the next segment using the adapted value and lookahead scores, then continues from the updated prefix.

For candidate features $\phi(s,a)$, the adapted value is

$$
V_t(s,a)=\phi(s,a)^\top(w_0+\delta_t),
$$

where $w_0$ is the initial critic and $\delta_t$ is reset for each new prompt. OVAL inference uses the released critic and does not require an external reward model. The supplied configuration uses 12 candidate segments per step and two lookahead continuations per candidate.

## Repository structure

```text
jftta/
├── README.md
├── exp/
│   ├── run.py                    OVAL inference entry point
│   ├── requirements.txt
│   ├── configs/oval.json         Generation and search settings
│   ├── data/eval500.jsonl        Complete 500-prompt input set
│   └── src/
│       ├── oval.py              Segment search and FTRL updates
│       ├── value_head/          Critic loading and hidden-state features
│       ├── models.py            Model loading and generation
│       ├── search_common.py     Shared search utilities
│       ├── baselines/           Baseline inference implementations
│       └── rewards.py           Reward-model utilities
├── annotation/
├── budget_cmp/
├── plot/
├── rr2/
└── runtime/
```

`exp/` contains the released inference experiment. The other top-level directories retain the supporting analysis, plotting, and benchmark material. Model and critic checkpoints are hosted separately on Hugging Face.

## Run the 500-prompt experiment

Clone the repository and install the dependencies in an environment with CUDA-enabled PyTorch:

```bash
git clone https://github.com/lakjx/jftta.git
cd jftta
pip install -r exp/requirements.txt
```

Checkpoint location: [`aaaa12138/test1/ckpt0` on Hugging Face](https://huggingface.co/aaaa12138/test1/tree/main/ckpt0). The runner expects this layout:

```text
ckpt0/
├── model/                       Complete model and tokenizer
└── critics/
    ├── helpfulness.pt
    ├── honesty.pt
    ├── instruction_following.pt
    └── truthfulness.pt
```

The policy model already contains the merged base and LoRA weights, so inference loads a single complete model. The four dimension-named critic files use the same released ranking weights.

Run the experiment:

```bash
python exp/run.py \
  --checkpoint aaaa12138/test1 \
  --checkpoint-subdir ckpt0 \
  --dimension all \
  --device cuda:0 \
  --output-dir outputs/oval
```

This uses all 500 prompts in `exp/data/eval500.jsonl` and the settings in `exp/configs/oval.json`. It runs each dimension separately and writes `helpfulness.jsonl`, `honesty.jsonl`, `instruction_following.jsonl`, and `truthfulness.jsonl` under `outputs/oval/`, with one response per prompt.

Use `--dimension helpfulness`, `honesty`, `instruction_following`, or `truthfulness` to run only one dimension. Add `--resume` to the same command to continue an interrupted run. For downloaded checkpoints, replace the checkpoint options with `--checkpoint /path/to/ckpt0`.
