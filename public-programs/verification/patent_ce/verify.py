"""Independent standard-library audit of published CSVs and JSON statistics."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from core import METHODS, PROJECT, atomic_json, digest, load_json, stamp


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a,b,message):
    check(math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),message)


def independent_stats(items, method):
    counter=Counter((int(r['human_label']),int(r[method+'_prediction'])) for r in items)
    matrix=[[counter[y,p] for p in [-1,0,1]] for y in [-1,0,1]]
    n=len(items)
    correct=sum(counter[k,k] for k in [-1,0,1])
    f1s=[]
    supports=[]
    for k in [-1,0,1]:
        tp=counter[k,k]
        fp=sum(counter[y,k] for y in [-1,0,1] if y!=k)
        fn=sum(counter[k,p] for p in [-1,0,1] if p!=k)
        f1s.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
        supports.append(tp+fn)
    return dict(n=n,correct=correct,accuracy=correct/n,macro_f1=sum(f1s)/3,
                weighted_f1=sum(f*s for f,s in zip(f1s,supports))/n,confusion_matrix=matrix)


def verify(out):
    out=Path(out)
    result=load_json(out/'results.json')
    prepared=load_json(out/'pairs.json')
    with (out/'scores.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    check(len(rows)==1228,'CSV row count must be1228')
    originals={r['record_id']:r for r in prepared['records']}
    check(len({r['record_id'] for r in rows})==1228,'Duplicate CSV IDs')
    similarity=load_json(out/'similarity_scores.json')
    patclaim=load_json(out/'patclaim_scores.json')
    checks=0
    for row in rows:
        original=originals[row['record_id']]
        for key in ['gold_claim','A','B','split']:
            check(row[key]==original[key],f'Raw text/key mismatch:{row["record_id"]}:{key}')
        check(int(row['human_label'])==original['human_label'],'Human label mismatch')
        for k in ['a','b']:
            pid=hashlib.sha256(json.dumps([row['gold_claim'],row[k.upper()]],ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
            check(pid==row[f'pair_{k}_id'],'CSV pair hash mismatch')
            worker=similarity['scores'][pid]
            pc=patclaim['scores'][pid]
            check(int(row['bertscore_gold_tokens'])==worker['gold_tokens'],'Reference token count mismatch')
            check(int(row[f'bertscore_{k}_tokens'])==worker['candidate_tokens'],'Candidate token count mismatch')
            check((row['bertscore_gold_truncated']=='True')==worker['gold_truncated'],'Reference truncation mismatch')
            check((row[f'bertscore_{k}_truncated']=='True')==worker['candidate_truncated'],'Candidate truncation mismatch')
            check(int(row[f'patclaim_{k}_tokens'])==pc['pair_tokens'],'Pair token count mismatch')
            check((row[f'patclaim_{k}_truncated']=='True')==pc['truncated'],'Pair truncation mismatch')
            for method in METHODS:
                expected=worker[method] if method in METHODS[:4] else pc[method]
                close(row[f'{method}_score_{k}'],expected,'CSV score mismatch with worker')
        for method in METHODS:
            a,b=float(row[method+'_score_a']),float(row[method+'_score_b'])
            delta=a-b
            close(row[method+'_delta'],delta,'Delta mismatch')
            expected=0 if -0.0001<delta<0.0001 else 1 if delta>0 else -1
            check(int(row[method+'_prediction'])==expected,'Prediction mismatch at strict threshold')
            check((row[method+'_correct']=='True')==(expected==int(row['human_label'])),'Correctness mismatch')
            checks+=1
        check((row['bertscore_truncated']=='True')==any(int(row[k])>512 for k in ['bertscore_gold_tokens','bertscore_a_tokens','bertscore_b_tokens']),'BERT truncation mismatch')
        check((row['patclaim_truncated']=='True')==any(int(row[k])>2048 for k in ['patclaim_a_tokens','patclaim_b_tokens']),'Patclaim truncation mismatch')
    cohort_filters={
        'test':lambda r:r['split']=='test',
        'train':lambda r:r['split']=='train',
        'test_no_train_comparison':lambda r:r['split']=='test' and r['overlap_train_comparison']=='False',
        'test_no_gold_copy':lambda r:r['split']=='test' and r['contains_gold_copy']=='False',
        'train_deduplicated':lambda r:r['split']=='train' and r['duplicate_full_row']=='False',
        'test_bertscore_truncated':lambda r:r['split']=='test' and r['bertscore_truncated']=='True',
        'test_bertscore_untruncated':lambda r:r['split']=='test' and r['bertscore_truncated']=='False',
        'train_bertscore_truncated':lambda r:r['split']=='train' and r['bertscore_truncated']=='True',
        'train_bertscore_untruncated':lambda r:r['split']=='train' and r['bertscore_truncated']=='False',
    }
    for cohort,selector in cohort_filters.items():
        items=[r for r in rows if selector(r)]
        check(len(items)==result['cohorts'][cohort]['n'],'Cohort size mismatch:'+cohort)
        for method in METHODS:
            actual=independent_stats(items,method)
            expected=result['cohorts'][cohort]['results'][method]
            if cohort in ['train','test']:
                check(expected==result['results'][cohort][method],'Main/cohort JSON statistics disagree')
            for key in ['n','correct','confusion_matrix']:
                check(actual[key]==expected[key],f'Stat mismatch {cohort}:{method}:{key}')
            for key in ['accuracy','macro_f1','weighted_f1']:
                close(actual[key],expected[key],f'Stat mismatch {cohort}:{method}:{key}')
    with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        summary=list(csv.DictReader(stream))
    check(len(summary)==len(cohort_filters)*len(METHODS),'Summary CSV count mismatch')
    check(len({(r['cohort'],r['method']) for r in summary})==len(summary),'Summary CSV duplicate keys')
    for r in summary:
        s=result['cohorts'][r['cohort']]['results'][r['method']]
        for key in ['n','correct','accuracy','error_rate','macro_f1','weighted_f1','tie_recall','non_tie_accuracy','reverse_rate','continuous_tau','continuous_rho','discrete_tau','discrete_rho']:
            if s[key] is None:
                check(r[key]=='','Summary undefined value is not blank')
            else:
                close(r[key],s[key],'Summary CSV value mismatch:'+key)
        for metric in ['accuracy','macro_f1']:
            close(r[metric+'_ci_low'],s['ci'][metric][0],'Summary lower CI mismatch')
            close(r[metric+'_ci_high'],s['ci'][metric][1],'Summary upper CI mismatch')
    with (out/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        tall=list(csv.DictReader(stream))
    check(len(tall)==1228*6,'Tall CSV row count mismatch')
    wide_by_id={r['record_id']:r for r in rows}
    seen=set()
    for r in tall:
        key=(r['record_id'],r['method'])
        check(key not in seen,'Duplicate tall comparison')
        seen.add(key)
        for field in ['score_a','score_b','delta','prediction','correct']:
            check(r[field]==wide_by_id[r['record_id']][r['method']+'_'+field],'Tall/wide disagreement')
    for name,expected in prepared['source_hashes'].items():
        check(digest(PROJECT/'data'/'Patent-CE'/name)==expected,'Original source changed:'+name)
    check(len(similarity['scores'])==224 and len(patclaim['scores'])==224,'Unique pair count mismatch')
    check(result['audit']['test']['rows']==184 and result['audit']['train']['rows']==1044,'Split counts mismatch')
    check(result['audit']['truncation']['test']['bertscore_rows']==151,'BERT truncation audit changed')
    check(result['audit']['truncation']['test']['patclaim_rows']==3,'Patclaim truncation audit changed')
    check(result['baselines']['majority']['label']==-1,'Majority label mismatch')
    report=dict(status='passed',verified_at_taipei=stamp(),raw_records=1228,unique_pairs=224,
                independently_verified_preference_decisions=checks,cohorts_verified=len(cohort_filters),
                methods_verified=6,raw_text_exact_match=True,source_hashes_unchanged=True,
                tall_wide_agreement=True,confusion_accuracy_macro_weighted_f1_recomputed=True,
                artifact_hashes={name:digest(out/name) for name in ['scores.csv','predictions.csv','summary.csv','pairs.json','similarity_scores.json','patclaim_scores.json']})
    result['verification']=report
    atomic_json(out/'verification.json',report)
    atomic_json(out/'results.json',result)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    verify(args.output_dir)
