"""Package reviewed reports, auditable results and reproducible source files."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import zipfile
from core import PROJECT, atomic_json, digest, load_json, stamp


def package(out):
    out=Path(out).resolve()
    if load_json(out/'verification.json')['status']!='passed':
        raise ValueError('Numerical verification has not passed')
    qa=load_json(out/'qa_review.json')
    if qa['status']!='passed':
        raise ValueError('Every final report page must pass visual review before delivery')
    for name in ['similarity_resume_verification.json','patclaim_resume_verification.json']:
        if load_json(out/name)['status']!='passed':
            raise ValueError('Actual checkpoint resume proof has not passed: '+name)
    if not list(out.glob('*.docx')) or not list(out.glob('*.pdf')):
        raise FileNotFoundError('Both Word and PDF reports are required')
    if digest(next(out.glob('*.docx')))!=qa['docx_sha256'] or digest(next(out.glob('*.pdf')))!=qa['pdf_sha256']:
        raise ValueError('Final report changed after visual review')
    source_dir=Path(__file__).resolve().parent
    sources=list(source_dir.glob('*.py'))+[source_dir/'README.txt',source_dir.parent/'run_patent_ce_evaluation.py',
        PROJECT/'LLM4DPCG'/'requirements-local.lock.txt',PROJECT/'public-programs'/'verification'/'requirements-offline.txt']
    sources.extend((PROJECT/'data'/'Patent-CE').glob('*.json'))
    sources.append(PROJECT/'data'/'Patent-CE'/'UPSTREAM_README.md')
    outputs=[p for p in out.iterdir() if p.is_file() and p.suffix.lower() in ['.docx','.pdf','.csv','.json','.jsonl','.txt'] and p.name not in ['delivery_manifest.json','archive_sha256.txt']]
    outputs.extend((out/'figures').glob('*.png'))
    paths=sorted(set(sources+outputs),key=lambda p:str(p).lower())
    manifest=dict(created_at_taipei=stamp(),status='verified_and_reviewed',
                  contents={p.relative_to(PROJECT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths},
                  excluded='Model weights, virtual environments and intermediate page PNGs are not included',
                  test_results='18 unit tests passed; both model checkpoints passed actual zero-inference resume checks',
                  readme='public-programs/verification/patent_ce/README.txt')
    manifest_path=out/'delivery_manifest.json'
    atomic_json(manifest_path,manifest)
    paths.append(manifest_path)
    target=out/'Patent_CE_完整評估報告與重跑資料.zip'
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:
            archive.write(path,path.relative_to(PROJECT).as_posix())
    os.replace(temporary,target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    (out/'archive_sha256.txt').write_text(digest(target)+'  '+target.name+'\n',encoding='utf-8')
    print(str(target),flush=True)
    print(f'{len(paths)} files; {target.stat().st_size} bytes; ZIP CRC passed',flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    package(args.output_dir)
