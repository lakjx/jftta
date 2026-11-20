"""Measure generation latency, full FTRL-update time, and optimizer state size."""
from __future__ import annotations

import argparse
import copy
import functools
import gc
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE/'source'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import transformers
import isolated_ftrl as ftrl
from src.common import set_seed, stable_prompt_seed
from src.models import generate_texts, load_lm, render_chat
from src.value_head.judge import CandidateHiddenJudge


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def cfg_load():
    cfg = json.loads((HERE / 'config.json').read_text())
    cfg['models']['llama'] = str(REPO / cfg['models']['llama'])
    cfg['paths']['sft_adapter'] = str(REPO / cfg['paths']['sft_adapter'])
    cfg['judge']['probe_path'] = str(REPO / cfg['judge']['probe_path'])
    cfg['evaluation']['prompts'] = str(REPO / cfg['evaluation']['prompts'])
    for key in cfg.get('cache', {}):
        cfg['cache'][key] = str(HERE / 'cache' / key)
    return cfg


class Meter:
    def __init__(self):
        self.original_gen=ftrl.generate_texts
        self.original_la=ftrl.generate_ftrl_lookahead_records
        self.original_gradient=ftrl.ftrl_pairwise_gradient
        self.original_prox=ftrl.ftrl_proximal_delta
        self.reset()
        self.micro=[]
        ftrl.generate_texts=self.generate
        ftrl.generate_ftrl_lookahead_records=self.lookahead
        ftrl.ftrl_pairwise_gradient=self.gradient
        ftrl.ftrl_proximal_delta=self.prox

    def reset(self):
        self.calls=[]
        self.times={'generation_seconds':0.,'features_seconds':0.,'gradient_seconds':0.,'prox_seconds':0.}
        self.in_la=False

    def generate(self,*args,**kw):
        torch.cuda.synchronize(); start=time.perf_counter()
        rows=self.original_gen(*args,**kw)
        torch.cuda.synchronize(); sec=time.perf_counter()-start
        self.times['generation_seconds']+=sec
        self.calls.append({'kind':'lookahead' if self.in_la else 'actions',
                           'n':len(rows),'generated_tokens':sum(int(r['raw_token_count'])+int(r['stop_reason']=='stop_token') for r in rows),
                           'seconds':sec})
        return rows

    def lookahead(self,*args,**kw):
        if bool(args[0]['search'].get('runtime_fixed_fast',False)):
            return []
        self.in_la=True
        try: return self.original_la(*args,**kw)
        finally: self.in_la=False

    def gradient(self,*args,**kw):
        start=time.perf_counter(); out=self.original_gradient(*args,**kw)
        self.times['gradient_seconds']+=time.perf_counter()-start
        if len(self.micro)<64 and out[1]>0:
            self.micro.append({'features':args[0].copy(),'targets':list(args[1]),'weight':args[2].copy(),
                               'margin':args[3],'max_pairs':args[4]})
        return out

    def prox(self,*args,**kw):
        start=time.perf_counter(); out=self.original_prox(*args,**kw)
        self.times['prox_seconds']+=time.perf_counter()-start
        return out

    def feature_wrapper(self,fn):
        @functools.wraps(fn)
        def wrapped(*a,**kw):
            torch.cuda.synchronize(); start=time.perf_counter()
            out=fn(*a,**kw)
            torch.cuda.synchronize(); self.times['features_seconds']+=time.perf_counter()-start
            return out
        return wrapped

    def microbenchmark(self,cfg):
        # Uses actual action features and rollout targets, including all feature dimensions.
        samples=[]; s=cfg['search']
        for item in self.micro:
            w=item['weight']; z=np.zeros_like(w); n=np.zeros_like(w)
            durations=[]
            for rep in range(100):
                start=time.perf_counter()
                grad,pairs=self.original_gradient(item['features'],item['targets'],w,item['margin'],item['max_pairs'])
                zz=z+grad; nn=n+grad*grad
                self.original_prox(zz,nn,alpha=s['ftrl_alpha'],beta=s['ftrl_beta'],l1=s['ftrl_l1'],l2=s['ftrl_l2'],max_delta_norm=s['ftrl_max_delta_norm'])
                durations.append(1000*(time.perf_counter()-start))
            samples.append({'d':len(w),'actions':len(item['features']),'pairs':pairs,'median_ms':statistics.median(durations),'p95_ms':float(np.quantile(durations,.95))})
        return {'actual_feature_updates':samples,'repeats_per_update':100,'mean_update_median_ms':statistics.mean(x['median_ms'] for x in samples) if samples else None,
                'extra_state_bytes':3*len(self.micro[0]['weight'])*4 if self.micro else None,
                'description':'Actual gradients and features replayed on CPU; full gradient + accumulation + FTRL proximal computation; no LM backward pass.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=16)
    args=ap.parse_args(); torch.set_num_threads(6); device=torch.device('cuda:0')
    if os.environ.get('OVAL_GPU_MEM_FRACTION'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['OVAL_GPU_MEM_FRACTION']))
    cfg=cfg_load(); dump(HERE/'run_config.json',cfg)
    source=Path(cfg['evaluation']['prompts'])
    prompts=[json.loads(x) for x in source.read_text().splitlines()]
    random.Random(20261107).shuffle(prompts); prompts=prompts[:args.limit]
    dump(HERE/'cohort.json',{'selection':'Seed20261107 simple random sample from all 518 HelpSteer2 validation prompts, before new generation',
         'prompt_ids':[p['prompt_id'] for p in prompts]})
    out=HERE/'results'; out.mkdir(exist_ok=True)
    dump(HERE/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
         'pid':os.getpid(),'timing':'CUDA-synchronized generation and method selection, excludes model loading/evaluation; feature/gen timers synchronize boundaries',
         'seed':42,'namespace':cfg['search']['seed_namespace'],'target_blend':cfg['search']['ftrl_target_score_blend'],
         'variants':{'OVAL':'Original FTRL with across-step accumulation and target blend0.5','Reset':'Only FTRL sufficient statistics and delta zeroed at start of each step, blend0.5 unchanged',
                     'Fixed':'Frozen critic action ranking; no lookahead and no update; same policy and action count8',
                     'Direct':'Frozen-critic lookahead backup ranking; same action/lookahead configuration; first8 prompts'}})
    print('Loading policy',flush=True)
    tok,model=load_lm(cfg,device,adapter_path=cfg['paths']['sft_adapter'])
    original_forward=model.forward
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    parameters = inspect.signature(base_model.forward).parameters
    logits_argument = next(
        (name for name in ['logits_to_keep', 'num_logits_to_keep'] if name in parameters),
        None,
    )
    @functools.wraps(original_forward)
    def compact_forward(*a,**kw):
        if kw.get('use_cache') is False and logits_argument is not None:
            kw.setdefault(logits_argument, 1)
        return original_forward(*a,**kw)
    model.forward=compact_forward
    judge=CandidateHiddenJudge(cfg,device,tok,model); meter=Meter()
    judge.feature_matrix=meter.feature_wrapper(judge.feature_matrix)
    set_seed(42); chat=render_chat(tok,'','Write one sentence about science.',None,True)
    generate_texts(cfg,tok,model,device,[chat],8,max_new_tokens=16)
    judge.score('Write one sentence about science.','',['Science studies nature.']*8)
    for pi,prompt in enumerate(prompts):
        path=out/(prompt['prompt_id']+'.json')
        saved=json.loads(path.read_text()) if path.exists() else {'prompt':prompt,'methods':{}}
        order=['OVAL','Reset','Fixed']+(['Direct'] if pi<8 else [])
        order=order[pi%len(order):]+order[:pi%len(order)]
        for mode in order:
            if mode in saved['methods']: continue
            c=copy.deepcopy(cfg)
            if mode=='Reset': c['search']['runtime_step_reset']=True
            if mode=='Fixed': c['search'].update(ftrl_alpha=0.,ftrl_lookahead_per_action=0,runtime_fixed_fast=True)
            if mode=='Direct': c['search'].update(ftrl_alpha=0.,runtime_direct_backup=True)
            set_seed(stable_prompt_seed(42,prompt['prompt_id'],c['search']['seed_namespace']))
            meter.reset(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start=time.perf_counter()
            row=ftrl.generate_ftrl_mpc(c,tok,model,judge,device,prompt)
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            row.update(e2e_seconds=seconds,total_generated_tokens=sum(x['generated_tokens'] for x in meter.calls),
                       generation_calls=copy.deepcopy(meter.calls),timings=copy.deepcopy(meter.times),
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       update_state_bytes=3*judge.weight.nbytes)
            saved['methods'][mode]=row; dump(path,saved)
            if mode==order[-1]: dump(HERE/'update_microbenchmark.json',meter.microbenchmark(cfg))
            print(json.dumps({'prompt':pi+1,'id':prompt['prompt_id'],'method':mode,'seconds':round(seconds,2),
                  'tokens':row['total_generated_tokens'],'update_ms':1000*(row['timings']['gradient_seconds']+row['timings']['prox_seconds'])}),flush=True)
    judge.close(); model.cpu(); del judge,model,tok,original_forward,base_model; gc.collect(); torch.cuda.empty_cache()
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
