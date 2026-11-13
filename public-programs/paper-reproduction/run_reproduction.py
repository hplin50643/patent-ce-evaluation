"""Replay the released model, then reconstruct training without fitting to test targets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, LongformerConfig, LongformerModel

from author_modeling import ContrastiveModel
from protocol import load_json, atomic_json, sha, stamp, loss_tensor, predictions, evaluate_rows

STOP=False
class StopRun(Exception):pass
def signal_stop(*_):
    global STOP
    STOP=True

def check_stop(out,deadline):
    if STOP or (out/'STOP').exists() or time.time()>=deadline:raise StopRun('Stopped at fixed deadline/interruption')

def state(path,stage,**kwargs):atomic_json(path,dict(stage=stage,updated_at=stamp(),**kwargs))

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False

def fingerprint(out):
    files=['experiment_config.json','dataset.json','split_manifest.json']
    hashes={name:sha(out/name) for name in files}
    hashes.update({name:sha(Path(__file__).parent/name) for name in ['run_reproduction.py','author_modeling.py','protocol.py','prepare_original.py']})
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),hashes

def initialize(config,seed):
    seed_all(seed)
    spec=config['base_model']
    base=LongformerModel.from_pretrained(spec['model_id'],revision=spec['revision'],attn_implementation='eager')
    model=ContrastiveModel(base.config)
    model.bert.load_state_dict(base.state_dict(),strict=True)
    del base
    model.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model.float().cuda()

def encode(rows,tokenizer,max_length):
    cache={}
    for row in rows:
        for key in ['A','B']:
            pair=(row['gold_claim'],row[key])
            if pair not in cache:
                full=tokenizer(*pair,truncation=False,return_token_type_ids=False)
                if len(full['input_ids'])>max_length:raise ValueError('Paper says no truncation; original pair exceeds cap')
                cache[pair]=full
    return cache

def batch(parts,tokenizer):
    return {k:v.cuda() for k,v in tokenizer.pad(parts,padding=True,pad_to_multiple_of=512,return_tensors='pt').items() if k in ['input_ids','attention_mask']}

def score(model,rows,cache,tokenizer,out,deadline):
    model.eval();scores={}
    with torch.inference_mode():
        for row in rows:
            for key in ['A','B']:
                pair=(row['gold_claim'],row[key])
                if pair not in scores:
                    check_stop(out,deadline)
                    value=float(model.compute_score(**batch([cache[pair]],tokenizer)).reshape(-1)[0])
                    if not math.isfinite(value):raise ValueError('Nonfinite score')
                    scores[pair]=value
    return predictions(rows,scores)

def row_loss(rows,settings):
    return float(loss_tensor(torch.tensor([r['score_a'] for r in rows]),torch.tensor([r['score_b'] for r in rows]),
                            torch.tensor([r['label'] for r in rows]),settings['margin'],settings['tie_tolerance']))

def export(model,tokenizer,path,metadata):
    model.save_pretrained(path,safe_serialization=True)
    tokenizer.save_pretrained(path)
    atomic_json(path/'reproduction_metadata.json',metadata)

def replay_author(out,config,rows,deadline,fp):
    folder=out/'author';folder.mkdir(exist_ok=True)
    status=folder/'worker_state.json'
    state(status,'loading_released_checkpoint')
    spec=config['author_checkpoint']
    snapshot=Path(snapshot_download(spec['model_id'],revision=spec['revision']))
    if sha(snapshot/'model.safetensors')!=spec['weights_sha256']:raise ValueError('Author checkpoint SHA256 mismatch')
    model=ContrastiveModel.from_pretrained(snapshot,local_files_only=True).float().cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    pairs=list(dict.fromkeys((r['gold_claim'],r[k]) for r in rows for k in ['A','B']))
    pair_lengths={pair:len(tokenizer(*pair,truncation=False,return_token_type_ids=False)['input_ids']) for pair in pairs}
    original_test=[r for r in rows if r['source_split']=='test']
    for length in config['evaluation']['author_lengths']:
        started=time.monotonic();scores={}
        for i,pair in enumerate(pairs):
            check_stop(out,deadline)
            value=model.score_pair(*pair,tokenizer,'cuda',max_length=length)
            if not math.isfinite(value):raise ValueError('Nonfinite released checkpoint score')
            scores[pair]=value
            if i%20==0:state(status,'score_released_checkpoint',max_length=length,completed_pairs=i,total_pairs=len(pairs))
        pred=predictions(original_test,scores,dict(max_length=length,checkpoint='released_author'))
        atomic_json(folder/f'predictions_{length}.json',pred)
        atomic_json(folder/f'result_{length}.json',dict(status='complete',fingerprint=fp,model=spec,
                    max_length=length,padding='max_length',precision='float32',model_eval=True,
                    max_original_pair_tokens=max(pair_lengths.values()),
                    truncated_unique_pairs=sum(n>length for n in pair_lengths.values()),
                    elapsed_seconds=time.monotonic()-started,metrics=evaluate_rows(pred),
                    inference_code_sha256=sha(Path(__file__).parent/'author_modeling.py'),
                    test_used_for_training_or_selection=False))
    state(status,'complete')

def preflight(out,config,rows,deadline,fp):
    model=initialize(config,config['training']['primary_seed'])
    spec=config['base_model'];tok=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'])
    cache=encode(rows,tok,config['training']['max_length'])
    longest=max(cache.values(),key=lambda p:len(p['input_ids']))
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    inputs=batch([longest]*8,tok)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    before=model.fc.weight.detach().clone()
    opt.zero_grad(set_to_none=True)
    values=model.compute_score(**inputs).reshape(-1)
    loss=(values-.75).square().mean();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
    if torch.equal(before,model.fc.weight):raise ValueError('Preflight did not update')
    result=dict(status='passed',fingerprint=fp,precision='float32',batch_comparisons=4,batch_sequences=8,
                longest_pair_tokens=len(longest['input_ids']),padded_tokens=inputs['input_ids'].shape[1],
                elapsed_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                gradient_checkpointing=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    path=out/'capability_test_only'
    model.eval()
    with torch.inference_mode():expected=model.compute_score(**inputs).detach().cpu()
    export(model,tok,path,dict(capability_test_only=True))
    del opt,model;torch.cuda.empty_cache()
    restored=ContrastiveModel.from_pretrained(path,local_files_only=True).float().cuda().eval()
    with torch.inference_mode():actual=restored.compute_score(**inputs).detach().cpu()
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    result['save_reload_max_abs_error']=float((actual-expected).abs().max())
    atomic_json(out/'preflight.json',result)
    print(json.dumps(result),flush=True)

def save_last(path,model,opt,progress):
    payload=dict(progress,model=model.state_dict(),optimizer=opt.state_dict(),rng_python=random.getstate(),
                 rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
    temp=path.with_name(path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)

def training(out,config,rows,manifest,seed,deadline,fp):
    settings=config['training'];folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
    status=folder/'worker_state.json';last=folder/'last_state.pt'
    if (folder/'result.json').exists():
        if load_json(folder/'result.json')['fingerprint']!=fp:raise ValueError('Completed source mismatch')
        return
    if load_json(out/'preflight.json')['status']!='passed':raise ValueError('FP32 preflight required')
    state(status,'initializing',seed=seed)
    model=initialize(config,seed)
    spec=config['base_model'];tok=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['revision'])
    cache=encode(rows,tok,settings['max_length'])
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'],betas=tuple(settings['betas']),eps=settings['adam_epsilon'])
    lookup={r['comparison_id']:r for r in rows};split=manifest['folds'][0]
    parts={k:[lookup[i] for i in split[k]['comparison_ids']] for k in ['train','validation','test']}
    progress=dict(fingerprint=fp,epoch=0,next_row=0,updates=0,best=None,history=[],elapsed_seconds=0.0,peak_vram_bytes=0)
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['fingerprint']!=fp:raise ValueError('Resume source/data mismatch')
        model.load_state_dict(checkpoint.pop('model'));opt.load_state_dict(checkpoint.pop('optimizer'))
        random.setstate(checkpoint.pop('rng_python'));np.random.set_state(checkpoint.pop('rng_numpy'))
        torch.set_rng_state(checkpoint.pop('rng_torch'));torch.cuda.set_rng_state_all(checkpoint.pop('rng_cuda'))
        progress=checkpoint
    started=time.monotonic();prior=progress['elapsed_seconds'];torch.cuda.reset_peak_memory_stats()
    def save():
        progress['elapsed_seconds']=prior+time.monotonic()-started
        progress['peak_vram_bytes']=max(progress['peak_vram_bytes'],torch.cuda.max_memory_allocated())
        save_last(last,model,opt,progress)
    try:
        for epoch in range(progress['epoch'],settings['epochs']):
            progress['epoch']=epoch;order=list(parts['train']);random.Random(seed+epoch).shuffle(order)
            model.train()
            for begin in range(progress['next_row'],len(order),settings['batch_size_comparisons']):
                check_stop(out,deadline)
                chunk=order[begin:begin+settings['batch_size_comparisons']];n=len(chunk)
                inputs=batch([cache[(r['gold_claim'],r[k])] for k in ['A','B'] for r in chunk],tok)
                opt.zero_grad(set_to_none=True)
                values=model.compute_score(**inputs).reshape(-1)
                loss=loss_tensor(values[:n],values[n:],torch.tensor([r['label'] for r in chunk],device='cuda'),settings['margin'],settings['tie_tolerance'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True)
                progress['next_row']=begin+n;progress['updates']+=1
                if progress['updates']%10==0:state(status,'train',seed=seed,epoch=epoch+1,completed_rows=begin+n,total_rows=len(order),updates=progress['updates'],loss=float(loss.detach()),elapsed_seconds=prior+time.monotonic()-started)
                if progress['updates']%config['runtime']['checkpoint_every_batches']==0 or STOP:save()
            save();check_stop(out,deadline)
            state(status,'validation',seed=seed,epoch=epoch+1)
            val=score(model,parts['validation'],cache,tok,out,deadline)
            vl=row_loss(val,settings)
            metrics=evaluate_rows(val)
            record=dict(epoch=epoch+1,validation_loss=vl,validation_accuracy=metrics['accuracy'],validation_continuous_tau=metrics['continuous']['kendall_tau_b'])
            progress['history'].append(record)
            if progress['best'] is None or vl<progress['best']['validation_loss']:
                progress['best']=record
                export(model,tok,folder/'validation_best',dict(seed=seed,epoch=epoch+1,selection='minimum validation loss',fingerprint=fp,settings=settings))
            atomic_json(folder/'history.json',progress['history'])
            print(json.dumps(dict(seed=seed,**record)),flush=True)
            progress['epoch']=epoch+1;progress['next_row']=0;save()
        export(model,tok,folder/'final',dict(seed=seed,epoch=10,selection='prespecified final epoch',fingerprint=fp,settings=settings))
        del opt,model;torch.cuda.empty_cache()
        result=dict(status='complete',seed=seed,primary=seed==settings['primary_seed'],fingerprint=fp,
                    assumptions=config['assumptions_not_disclosed_by_authors'],history=progress['history'],
                    elapsed_seconds=progress['elapsed_seconds'],peak_vram_bytes=progress['peak_vram_bytes'],checkpoints={})
        for checkpoint_name in ['final','validation_best']:
            check_stop(out,deadline)
            state(status,'test_prespecified_checkpoint',seed=seed,checkpoint=checkpoint_name)
            reloaded=ContrastiveModel.from_pretrained(folder/checkpoint_name,local_files_only=True).float().cuda().eval()
            test=score(reloaded,parts['test'],cache,tok,out,deadline)
            atomic_json(folder/f'{checkpoint_name}_test_predictions.json',test)
            result['checkpoints'][checkpoint_name]=dict(metadata=load_json(folder/checkpoint_name/'reproduction_metadata.json'),
                metrics=evaluate_rows(test),files={str(p.relative_to(folder/checkpoint_name)):sha(p) for p in (folder/checkpoint_name).rglob('*') if p.is_file()})
            del reloaded;torch.cuda.empty_cache()
        atomic_json(folder/'result.json',result);state(status,'complete',seed=seed)
    except StopRun:
        if 'opt' in locals():save()
        state(status,'stopped',seed=seed,epoch=progress['epoch'],next_row=progress['next_row']);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['author','preflight','train'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--deadline-utc',required=True);a=p.parse_args();out=a.output_dir.resolve()
    deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    config=load_json(out/'experiment_config.json');rows=load_json(out/'dataset.json');manifest=load_json(out/'split_manifest.json')
    torch.set_num_threads(config['runtime']['torch_threads'])
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,0):raise RuntimeError('Expected V100 CUDA environment')
    fp,hashes=fingerprint(out);atomic_json(out/'source_fingerprint.json',dict(fingerprint=fp,files=hashes))
    try:
        check_stop(out,deadline)
        if a.stage=='author':replay_author(out,config,rows,deadline,fp)
        elif a.stage=='preflight':preflight(out,config,rows,deadline,fp)
        else:
            if a.seed not in config['training']['seeds']:raise ValueError('Undeclared seed')
            training(out,config,rows,manifest,a.seed,deadline,fp)
    except StopRun as e:print(str(e),flush=True);return 124
    except Exception as e:
        atomic_json(out/f'failure_{a.stage}_{a.seed}.json',dict(stage=a.stage,seed=a.seed,error=str(e),traceback=traceback.format_exc(),recorded_at=stamp()))
        raise
    return 0

if __name__=='__main__':sys.exit(main())
