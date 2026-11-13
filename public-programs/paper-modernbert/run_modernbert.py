"""Train the ModernBERT replacement against the locked original-split control."""
from __future__ import annotations

import argparse
from datetime import datetime
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
from transformers import AutoTokenizer, ModernBertModel

from model import ModernContrastiveModel
from protocol import atomic_json, evaluate_rows, load_json, loss_tensor, predictions, sha, stamp

STOP = False


class StopRun(Exception):
    pass


def signal_stop(*_):
    global STOP
    STOP = True


def check_stop(out, deadline):
    if STOP or (out / 'STOP').exists() or time.time() >= deadline:
        raise StopRun('Stopped at fixed deadline/interruption')


def state(path, stage, **kwargs):
    atomic_json(path, dict(stage=stage, updated_at=stamp(), **kwargs))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def fingerprint(out):
    names = ['experiment_config.json', 'dataset.json', 'split_manifest.json', 'baseline_lock.json']
    hashes = {name: sha(out / name) for name in names}
    for name in ['run_modernbert.py', 'model.py', 'protocol.py', 'prepare_inputs.py']:
        hashes[name] = sha(Path(__file__).parent / name)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def base_snapshot(config):
    spec = config['base_model']
    snapshot = Path(snapshot_download(spec['model_id'], revision=spec['revision'],
                                     allow_patterns=['config.json', 'model.safetensors']))
    return snapshot


def initialize(config, seed, attention, snapshot=None):
    seed_all(seed)
    snapshot = snapshot or base_snapshot(config)
    base, info = ModernBertModel.from_pretrained(
        snapshot, local_files_only=True, attn_implementation=attention,
        reference_compile=False, output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'Incomplete pretrained encoder: {info}')
    model = ModernContrastiveModel(base.config)
    model.bert.load_state_dict(base.state_dict(), strict=True)
    del base
    model.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model.float().cuda()


def encode(rows, tokenizer, max_length):
    cache, records = {}, []
    for row in rows:
        lengths = {}
        for key in ['A', 'B']:
            pair = (row['gold_claim'], row[key])
            if pair not in cache:
                full = tokenizer(*pair, truncation=False, return_token_type_ids=False)
                if len(full['input_ids']) > max_length:
                    raise ValueError(f'No truncation authorized: {row["comparison_id"]}/{key} '
                                     f'has {len(full["input_ids"])} tokens > {max_length}')
                cache[pair] = full
            lengths[key] = len(cache[pair]['input_ids'])
        records.append(dict(comparison_id=row['comparison_id'], source_split=row['source_split'],
                            source_row_index=row['source_row_index'], gold_id=row['gold_id'],
                            tokens_a=lengths['A'], tokens_b=lengths['B'], truncated_a=False, truncated_b=False))
    payload = dict(pair_format='native tokenizer(reference, candidate)', truncation=False,
                   max_length=max_length, padding='dynamic, multiple of 512', rows=records,
                   unique_pairs=len(cache), longest_pair_tokens=max(len(part['input_ids']) for part in cache.values()),
                   truncated_unique_pairs=0)
    return cache, payload


def batch(parts, tokenizer):
    return {key: value.cuda() for key, value in tokenizer.pad(
        parts, padding=True, pad_to_multiple_of=512, return_tensors='pt').items()
        if key in ['input_ids', 'attention_mask']}


def score(model, rows, cache, tokenizer, out, deadline):
    model.eval()
    scores = {}
    with torch.inference_mode():
        for row in rows:
            for key in ['A', 'B']:
                pair = (row['gold_claim'], row[key])
                if pair not in scores:
                    check_stop(out, deadline)
                    value = float(model.compute_score(**batch([cache[pair]], tokenizer)).reshape(-1)[0])
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite score')
                    scores[pair] = value
    return predictions(rows, scores)


def row_loss(rows, settings):
    return float(loss_tensor(torch.tensor([row['score_a'] for row in rows]),
                             torch.tensor([row['score_b'] for row in rows]),
                             torch.tensor([row['label'] for row in rows]),
                             settings['margin'], settings['tie_tolerance']))


def export(model, tokenizer, path, metadata):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    atomic_json(path / 'reproduction_metadata.json', metadata)


def preflight_attempt(out, config, cache, tokenizer, deadline, fp, attention, snapshot):
    check_stop(out, deadline)
    model = initialize(config, config['training']['primary_seed'], attention, snapshot)
    longest = max(cache.values(), key=lambda part: len(part['input_ids']))
    model.train()
    settings = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'], betas=tuple(settings['betas']),
                                 eps=settings['adam_epsilon'])
    inputs = batch([longest] * 8, tokenizer)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    before = model.fc.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    values = model.compute_score(**inputs).reshape(-1)
    loss = (values - .75).square().mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite capability-test loss')
    loss.backward()
    encoder_grad = any(parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                       for parameter in model.bert.parameters())
    pooler_grad = model.pooler.weight.grad is not None and bool(torch.any(model.pooler.weight.grad != 0))
    if not encoder_grad or not pooler_grad:
        raise ValueError('Encoder or new pooler did not receive nonzero gradients')
    torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, model.fc.weight):
        raise ValueError('Capability test did not update the scoring head')
    result = dict(status='passed', fingerprint=fp, precision='float32', batch_comparisons=4, batch_sequences=8,
                  attention_implementation=attention, reference_compile=False,
                  longest_pair_tokens=len(longest['input_ids']), padded_tokens=inputs['input_ids'].shape[1],
                  elapsed_seconds=time.monotonic() - started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                  gradient_checkpointing=True, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}),
                  trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                  encoder_nonzero_gradient=encoder_grad, new_pooler_nonzero_gradient=pooler_grad,
                  head_initialization='pooler normal(0,.02)/zero; fc torch Linear defaults')
    path = out / 'capability_test_only'
    model.eval()
    with torch.inference_mode():
        expected = model.compute_score(**inputs).detach().cpu()
    export(model, tokenizer, path, dict(capability_test_only=True, attention_implementation=attention))
    del optimizer, model
    torch.cuda.empty_cache()
    restored = ModernContrastiveModel.from_pretrained(path, local_files_only=True,
                attn_implementation=attention, reference_compile=False).float().cuda().eval()
    with torch.inference_mode():
        actual = restored.compute_score(**inputs).detach().cpu()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    result['save_reload_max_abs_error'] = float((actual - expected).abs().max())
    return result


def preflight(out, config, rows, deadline, fp):
    spec = config['base_model']
    snapshot = base_snapshot(config)
    provenance = dict(model_id=spec['model_id'], revision=spec['revision'],
                      files={name: sha(snapshot / name) for name in ['config.json', 'model.safetensors']})
    if spec.get('weights_sha256') and provenance['files']['model.safetensors'] != spec['weights_sha256']:
        raise ValueError('Base-model weight SHA256 mismatch')
    atomic_json(out / 'base_model_provenance.json', provenance)
    tokenizer = AutoTokenizer.from_pretrained(spec['model_id'], revision=spec['revision'], trust_remote_code=False)
    cache, tokenization = encode(rows, tokenizer, config['training']['max_length'])
    atomic_json(out / 'tokenization.json', tokenization)
    failures = []
    for attention in ['sdpa', 'eager']:
        try:
            result = preflight_attempt(out, config, cache, tokenizer, deadline, fp, attention, snapshot)
            result['tokenization_sha256'] = sha(out / 'tokenization.json')
            result['backend_fallbacks'] = failures
            atomic_json(out / 'preflight.json', result)
            print(json.dumps(result), flush=True)
            return
        except StopRun:
            raise
        except Exception as error:
            failures.append(dict(attention_implementation=attention, error_type=type(error).__name__,
                                 error=str(error), traceback=traceback.format_exc()))
            atomic_json(out / 'preflight_attempt_failures.json', failures)
            # FP32/batch/sequence limits are fixed. OOM is a failed capability
            # gate, not permission to reduce input length or change precision.
            if isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
                raise
            if attention == 'eager' or not isinstance(error, (RuntimeError, ValueError, NotImplementedError)):
                raise
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError('No compatible attention backend')


def save_last(path, model, optimizer, progress):
    payload = dict(progress, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                   rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def training(out, config, rows, manifest, seed, deadline, fp):
    settings = config['training']
    folder = out / f'seed_{seed}'
    folder.mkdir(exist_ok=True)
    status, last = folder / 'worker_state.json', folder / 'last_state.pt'
    if (folder / 'result.json').exists():
        if load_json(folder / 'result.json')['fingerprint'] != fp:
            raise ValueError('Completed source mismatch')
        return
    gate = load_json(out / 'preflight.json')
    if gate['status'] != 'passed' or gate['fingerprint'] != fp or gate['precision'] != 'float32':
        raise ValueError('Matching FP32 preflight required')
    attention = gate['attention_implementation']
    if attention not in ['sdpa', 'eager'] or gate['reference_compile'] is not False:
        raise ValueError('Unsupported capability-test backend')
    state(status, 'initializing', seed=seed, attention_implementation=attention)
    model = initialize(config, seed, attention)
    spec = config['base_model']
    tokenizer = AutoTokenizer.from_pretrained(spec['model_id'], revision=spec['revision'], trust_remote_code=False)
    cache, tokenization = encode(rows, tokenizer, settings['max_length'])
    if load_json(out / 'tokenization.json') != tokenization or sha(out / 'tokenization.json') != gate['tokenization_sha256']:
        raise ValueError('Formal training tokenization differs from capability test')
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'],
                                 betas=tuple(settings['betas']), eps=settings['adam_epsilon'])
    lookup, split = {row['comparison_id']: row for row in rows}, manifest['folds'][0]
    parts = {key: [lookup[identifier] for identifier in split[key]['comparison_ids']] for key in ['train', 'validation', 'test']}
    progress = dict(fingerprint=fp, epoch=0, next_row=0, updates=0, best=None, history=[], elapsed_seconds=0., peak_vram_bytes=0)
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['fingerprint'] != fp:
            raise ValueError('Resume source/data mismatch')
        model.load_state_dict(checkpoint.pop('model'))
        optimizer.load_state_dict(checkpoint.pop('optimizer'))
        random.setstate(checkpoint.pop('rng_python'))
        np.random.set_state(checkpoint.pop('rng_numpy'))
        torch.set_rng_state(checkpoint.pop('rng_torch'))
        torch.cuda.set_rng_state_all(checkpoint.pop('rng_cuda'))
        progress = checkpoint
    started, prior = time.monotonic(), progress['elapsed_seconds']
    torch.cuda.reset_peak_memory_stats()

    def save():
        progress['elapsed_seconds'] = prior + time.monotonic() - started
        progress['peak_vram_bytes'] = max(progress['peak_vram_bytes'], torch.cuda.max_memory_allocated())
        save_last(last, model, optimizer, progress)

    try:
        for epoch in range(progress['epoch'], settings['epochs']):
            progress['epoch'] = epoch
            order = list(parts['train'])
            random.Random(seed + epoch).shuffle(order)
            model.train()
            for begin in range(progress['next_row'], len(order), settings['batch_size_comparisons']):
                check_stop(out, deadline)
                chunk = order[begin:begin + settings['batch_size_comparisons']]
                n = len(chunk)
                inputs = batch([cache[(row['gold_claim'], row[key])] for key in ['A', 'B'] for row in chunk], tokenizer)
                optimizer.zero_grad(set_to_none=True)
                values = model.compute_score(**inputs).reshape(-1)
                loss = loss_tensor(values[:n], values[n:], torch.tensor([row['label'] for row in chunk], device='cuda'),
                                   settings['margin'], settings['tie_tolerance'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress['next_row'] = begin + n
                progress['updates'] += 1
                if progress['updates'] % 10 == 0:
                    state(status, 'train', seed=seed, epoch=epoch + 1, completed_rows=begin + n, total_rows=len(order),
                          updates=progress['updates'], loss=float(loss.detach()), elapsed_seconds=prior + time.monotonic() - started,
                          attention_implementation=attention)
                if progress['updates'] % config['runtime']['checkpoint_every_batches'] == 0 or STOP:
                    save()
            save()
            check_stop(out, deadline)
            state(status, 'validation', seed=seed, epoch=epoch + 1)
            validation = score(model, parts['validation'], cache, tokenizer, out, deadline)
            metrics = evaluate_rows(validation)
            record = dict(epoch=epoch + 1, validation_loss=row_loss(validation, settings),
                          validation_accuracy=metrics['accuracy'], validation_continuous_tau=metrics['continuous']['kendall_tau_b'])
            progress['history'].append(record)
            metadata = dict(seed=seed, epoch=epoch + 1, fingerprint=fp, settings=settings,
                            attention_implementation=attention, reference_compile=False, base_model=spec)
            if progress['best'] is None or record['validation_loss'] < progress['best']['validation_loss']:
                progress['best'] = record
                export(model, tokenizer, folder / 'validation_best', dict(metadata, selection='minimum validation loss'))
            atomic_json(folder / 'history.json', progress['history'])
            print(json.dumps(dict(seed=seed, **record)), flush=True)
            progress['epoch'], progress['next_row'] = epoch + 1, 0
            save()
        export(model, tokenizer, folder / 'final', dict(seed=seed, epoch=settings['epochs'],
            selection='prespecified final epoch', fingerprint=fp, settings=settings,
            attention_implementation=attention, reference_compile=False, base_model=spec))
        del optimizer, model
        torch.cuda.empty_cache()
        result = dict(status='complete', seed=seed, primary=seed == settings['primary_seed'], fingerprint=fp,
                      assumptions=config['assumptions_not_disclosed_by_authors'], history=progress['history'],
                      attention_implementation=attention, reference_compile=False,
                      elapsed_seconds=progress['elapsed_seconds'], peak_vram_bytes=progress['peak_vram_bytes'], checkpoints={})
        for selection in ['final', 'validation_best']:
            check_stop(out, deadline)
            state(status, 'test_prespecified_checkpoint', seed=seed, checkpoint=selection)
            reloaded = ModernContrastiveModel.from_pretrained(folder / selection, local_files_only=True,
                        attn_implementation=attention, reference_compile=False).float().cuda().eval()
            test = score(reloaded, parts['test'], cache, tokenizer, out, deadline)
            atomic_json(folder / f'{selection}_test_predictions.json', test)
            result['checkpoints'][selection] = dict(metadata=load_json(folder / selection / 'reproduction_metadata.json'),
                metrics=evaluate_rows(test), files={str(path.relative_to(folder / selection)): sha(path)
                for path in (folder / selection).rglob('*') if path.is_file()})
            del reloaded
            torch.cuda.empty_cache()
        atomic_json(folder / 'result.json', result)
        state(status, 'complete', seed=seed)
    except StopRun:
        if 'optimizer' in locals():
            save()
        state(status, 'stopped', seed=seed, epoch=progress['epoch'], next_row=progress['next_row'])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'train'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deadline-utc', required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir.resolve()
    deadline = datetime.fromisoformat(arguments.deadline_utc.replace('Z', '+00:00')).timestamp()
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    config, rows, manifest = [load_json(out / name) for name in ['experiment_config.json', 'dataset.json', 'split_manifest.json']]
    torch.set_num_threads(config['runtime']['torch_threads'])
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError('Expected V100 CUDA environment')
    fp, hashes = fingerprint(out)
    atomic_json(out / 'source_fingerprint.json', dict(fingerprint=fp, files=hashes))
    try:
        check_stop(out, deadline)
        if arguments.stage == 'preflight':
            preflight(out, config, rows, deadline, fp)
        else:
            if arguments.seed not in config['training']['seeds']:
                raise ValueError('Undeclared seed')
            training(out, config, rows, manifest, arguments.seed, deadline, fp)
    except StopRun as error:
        print(str(error), flush=True)
        return 124
    except Exception as error:
        atomic_json(out / f'failure_{arguments.stage}_{arguments.seed}.json',
                    dict(stage=arguments.stage, seed=arguments.seed, error=str(error),
                         traceback=traceback.format_exc(), recorded_at=stamp()))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
