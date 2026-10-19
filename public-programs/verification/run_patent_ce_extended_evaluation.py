"""Versioned thirteen-method Patent-CE evaluation coordinator."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from patent_ce_extended.core import PROJECT, LEGACY, NEW_METHODS, METHODS, prepare, wsl_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare','preflight','score','aggregate','verify','report','all'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path, default=LEGACY)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=NEW_METHODS)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and args.stage != 'score':
        parser.error('--smoke is only supported with --stage score; smoke results cannot be aggregated as a full evaluation')
    if args.threads < 1 or args.batch_size < 1 or args.bootstrap_replicates < 100:
        parser.error('Threads and batch size must be positive; bootstrap replicates must be at least 100')
    out = args.output_dir.resolve()
    assets = (args.assets or out / 'assets.json').resolve()
    folder = Path(__file__).resolve().parent / 'patent_ce_extended'
    py = '/home/agent/.venvs/patent-ce-13methods-py310/bin/python'
    stages = ['prepare','preflight','score','aggregate','verify','report'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print('Running ' + stage, flush=True)
        if stage == 'prepare':
            print(prepare(out, args.legacy_dir), flush=True)
            continue
        if stage == 'preflight':
            probe = subprocess.run(['wsl.exe','-d','Ubuntu-26.04','--','test','-x','/usr/bin/time'],check=False)
            if probe.returncode:
                parser.error('WSL GNU /usr/bin/time is required for supplemental resource measurement; install the time package before preflight')
            cmd = ['wsl.exe','-d','Ubuntu-26.04','--','bash',wsl_path(folder/'bootstrap.sh'),'--output-dir',wsl_path(out)]
        elif stage == 'score':
            # A full run passes every selected smoke check before any full scoring.
            for smoke_run in ([True,False] if args.stage == 'all' else [args.smoke]):
                for method in args.methods:
                    if method not in NEW_METHODS:
                        if not (out/(method+'_scores.json')).exists():
                            parser.error('Run prepare to import saved legacy scores before selecting '+method)
                        print('Using verified saved scores: '+method,flush=True)
                        continue
                    cmd = ['wsl.exe','-d','Ubuntu-26.04','--',py,wsl_path(folder/'worker.py'),
                           '--method',method,'--input',wsl_path(out/'pairs.json'),
                           '--output-dir',wsl_path(out),'--assets',wsl_path(assets),
                           '--threads',str(args.threads),'--batch-size',str(args.batch_size)]
                    if smoke_run:
                        cmd.append('--smoke')
                    subprocess.run(cmd, check=True)
                    if smoke_run and method == 'alignscore':
                        proof = out/'smoke/alignscore/alignscore_optimization_parity.json'
                        if not proof.is_file():
                            raise RuntimeError('Successful AlignScore smoke must emit its optimization parity proof')
                        shutil.copy2(proof,out/'alignscore_optimization_validation.json')
            if args.stage == 'all':
                # Independently verify actual minimum/median raw lengths without
                # changing the frozen worker or the complete result population.
                subprocess.run(['wsl.exe','-d','Ubuntu-26.04','--',py,
                    wsl_path(folder/'supplemental_validation.py'),
                    '--input',wsl_path(out/'pairs.json'),'--output-dir',wsl_path(out),
                    '--assets',wsl_path(assets),'--threads',str(args.threads),
                    '--batch-size',str(args.batch_size)],check=True)
            continue
        elif stage in ['aggregate','verify']:
            exe = PROJECT / 'LLM4DPCG/.venv/Scripts/python.exe' if stage == 'aggregate' else Path(sys.executable)
            cmd = [str(exe),str(folder/(stage+'.py')),'--output-dir',str(out)]
            if stage == 'aggregate':
                cmd += ['--bootstrap-replicates',str(args.bootstrap_replicates)]
        else:
            runtime = Path(r'C:\Users\agent\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
            cmd = [str(runtime), str(folder/'build_report.py'), '--output-dir',str(out)]
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
