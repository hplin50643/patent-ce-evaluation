"""Reproducible Patent-CE evaluation entry point (Windows coordinator)."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from patent_ce.core import PROJECT, prepare, load_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'similarity', 'patclaim', 'aggregate', 'verify', 'report', 'all'])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage in ['prepare', 'all']:
        print(prepare(out), flush=True)
    folder = Path(__file__).resolve().parent / 'patent_ce'
    stages = ['similarity', 'patclaim', 'aggregate', 'verify', 'report'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage == 'prepare':
            continue
        if stage == 'similarity':
            provenance_path=out/'model_provenance.json'
            if provenance_path.exists():
                model=load_json(provenance_path)['bertscore']
                snapshot=PROJECT/'LLM4DPCG'/'.cache'/'huggingface'/'hub'/'models--roberta-large'/'snapshots'/model['revision']
                for name,metadata in model['files'].items():
                    if digest(snapshot/name)!=metadata['sha256']:
                        raise ValueError('BERTScore cached model changed: '+name)
            cmd = [str(PROJECT / 'LLM4DPCG/.venv/Scripts/python.exe'), str(folder / 'similarity_worker.py'), '--input', str(out / 'pairs.json'), '--output-dir', str(out), '--threads', str(args.threads), '--batch-size', str(args.batch_size)]
        elif stage == 'patclaim':
            def wsl_path(path):
                path = str(Path(path).resolve())
                return '/mnt/' + path[0].lower() + path[2:].replace('\\', '/')
            cmd = ['wsl.exe', '-d', 'Ubuntu-26.04', '--', '/home/agent/.venvs/patent-programs-py312/bin/python', wsl_path(folder / 'patclaim_worker.py'), '--input', wsl_path(out / 'pairs.json'), '--output-dir', wsl_path(out), '--threads', str(args.threads)]
        elif stage == 'aggregate':
            cmd = [str(PROJECT / 'LLM4DPCG/.venv/Scripts/python.exe'), str(folder / 'aggregate.py'), '--output-dir', str(out), '--bootstrap-replicates', str(args.bootstrap_replicates)]
        elif stage == 'verify':
            cmd = [sys.executable, str(folder / 'verify.py'), '--output-dir', str(out)]
        elif stage == 'report':
            runtime = Path(r'C:\Users\agent\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
            cmd = [str(runtime), str(folder / 'build_report.py'), '--results', str(out / 'results.json'), '--output-dir', str(out)]
        print('Running ' + stage, flush=True)
        subprocess.run(cmd, check=True)
        if stage == 'report':
            documents = list(out.glob('*.docx'))
            if len(documents) != 1:
                raise RuntimeError('Expected one generated DOCX for PDF rendering')
            subprocess.run([str(runtime), str(folder / 'render_report.py'), str(documents[0]),
                            '--output_dir', str(out / 'qa'), '--emit_pdf'], check=True)
            shutil.copy2(out / 'qa' / (documents[0].stem + '.pdf'), out / (documents[0].stem + '.pdf'))


if __name__ == '__main__':
    main()
