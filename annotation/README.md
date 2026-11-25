##### Goal

- Run ArmoRM and GLM-5.2 consistency checks against the original dataset annotations
- Audit existing OVAL segment boundaries.

```bash
cd ./annotation
annotation_python=/home/xxx/anaconda3/envs/xxxxx/bin/python
```

##### ArmoRM

```bash
$annotation_python armorm.py \
  --data-dir ../original_annotation_validation/data \
  --model ../models/reward/ArmoRM-Llama3-8B-v0.1 \
  --output ../annotation_runs/armorm.jsonl
```

##### GLM-5.2

Set `ANTCHAT_API_KEY` in the environment. 

```bash
$annotation_python glm.py \
  --data-dir ../original_annotation_validation/data \
  --tasks ../original_annotation_validation/internal_glm/data/original_tasks.jsonl \
  --output ../annotation_runs/glm.jsonl
```

Use `--base-url`, `--model`, and `--expected-returned-model` to specify a different internal endpoint or model alias. Add `--check` to validate inputs without API calls.

##### Segment audit

```bash
$annotation_python segment_audit.py --project-root ..
```

Scoring commands save scores to `--output`, resume completed records automatically, and print annotation agreement statistics.
