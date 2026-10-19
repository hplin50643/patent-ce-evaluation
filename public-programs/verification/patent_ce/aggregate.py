"""Aggregate raw candidate scores into auditable three-way preference results."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import itertools
import math
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr

from core import METHODS, PRIMARY, LABELS, EPSILON, PROJECT, atomic_json, digest, load_json, preference, stamp

PAPER = {
    'bleu4': (.495, .420, .269, .305),
    'rouge1_f1': (.505, .429, .292, .332),
    'rougeL_f1': (.511, .434, .303, .344),
    'bertscore_f1': (.484, .418, .239, .268),
    'patclaimeval': (.565, .574, .477, .602),
}


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def correlation(delta, y):
    if len(set(delta)) < 2 or len(set(y)) < 2:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return finite_or_none(kendalltau(delta, y, variant='b').statistic), finite_or_none(spearmanr(delta, y).statistic)


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                q1=float(np.quantile(values, .25)), q3=float(np.quantile(values, .75)),
                minimum=float(values.min()), maximum=float(values.max()))


def classification(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (y + 1, pred + 1), 1)
    tp = np.diag(matrix)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) != 0)
    n = len(y)
    non_tie = y != 0
    reverse = non_tie & (pred == -y)
    return dict(n=n, correct=int(tp.sum()), accuracy=float(tp.sum() / n), error_rate=float(1 - tp.sum() / n),
                macro_f1=float(f1.mean()), weighted_f1=float(np.dot(f1, support) / n),
                balanced_accuracy=float(recall.mean()),
                classes={str(k):dict(precision=float(precision[k+1]),recall=float(recall[k+1]),f1=float(f1[k+1]),support=int(support[k+1]),predicted=int(predicted[k+1])) for k in [-1,0,1]},
                confusion_matrix=matrix.tolist(), confusion_label_order=[-1,0,1],
                tie_recall=float(recall[1]), human_tie_n=int(support[1]), predicted_tie_n=int(predicted[1]),
                non_tie_n=int(non_tie.sum()), non_tie_accuracy=float((y[non_tie] == pred[non_tie]).mean()) if non_tie.any() else None,
                reverse_count=int(reverse.sum()), reverse_rate=float(reverse.sum() / non_tie.sum()) if non_tie.any() else None)


def method_statistics(rows, method):
    y = [r['human_label'] for r in rows]
    pred = [r['methods'][method]['prediction'] for r in rows]
    values = classification(y, pred)
    a = [r['methods'][method]['score_a'] for r in rows]
    b = [r['methods'][method]['score_b'] for r in rows]
    delta = [r['methods'][method]['delta'] for r in rows]
    values['continuous_tau'], values['continuous_rho'] = correlation(delta, y)
    values['discrete_tau'], values['discrete_rho'] = correlation(pred, y)
    values['score_statistics'] = dict(A=describe(a), B=describe(b), delta=describe(delta), pooled_candidates=describe(a+b))
    return values


def cluster_bootstrap(rows, replicates=10000, seed=42):
    """Resample entire gold groups; same draws across methods permit paired CIs."""
    groups = sorted({row['gold_id'] for row in rows})
    group_index = {value: i for i, value in enumerate(groups)}
    g = len(groups)
    tables = np.zeros((len(METHODS), g, 3, 3), dtype=np.int64)
    for row in rows:
        for m, method in enumerate(METHODS):
            tables[m, group_index[row['gold_id']], row['human_label'] + 1, row['methods'][method]['prediction'] + 1] += 1
    draws = np.random.default_rng(seed).multinomial(g, np.full(g, 1/g), size=replicates)
    confusion = np.einsum('rg,mgij->rmij', draws, tables, optimize=True)
    true = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    diag = np.diagonal(confusion, axis1=-2, axis2=-1)
    f1 = np.divide(2 * diag, true + predicted, out=np.zeros_like(diag, dtype=float), where=(true + predicted) != 0)
    boot_accuracy = diag.sum(axis=-1) / confusion.sum(axis=(-1, -2))
    boot_macro = f1.mean(axis=-1)
    ci = {method:dict(accuracy=np.quantile(boot_accuracy[:,m],[.025,.975]).tolist(),
                     macro_f1=np.quantile(boot_macro[:,m],[.025,.975]).tolist(),
                     gold_groups=g,replicates=replicates) for m,method in enumerate(METHODS)}
    paired = []
    comparisons = list(itertools.combinations(range(5),2)) + [(5,4)]
    for a,b in comparisons:
        paired.append(dict(method_a=METHODS[a],method_b=METHODS[b],
                           ci_accuracy=np.quantile(boot_accuracy[:,a]-boot_accuracy[:,b],[.025,.975]).tolist(),
                           ci_macro_f1=np.quantile(boot_macro[:,a]-boot_macro[:,b],[.025,.975]).tolist()))
    return ci, paired


def join_rows(prepared, similarity, patclaim):
    ids = {p['pair_id'] for p in prepared['pairs']}
    if set(similarity['scores']) != ids or set(patclaim['scores']) != ids:
        raise ValueError('Worker outputs do not exactly cover prepared pairs')
    result = []
    for original in prepared['records']:
        row = dict(original, methods={})
        a_id, b_id = row['pair_a_id'], row['pair_b_id']
        sa, sb = similarity['scores'][a_id], similarity['scores'][b_id]
        pa, pb = patclaim['scores'][a_id], patclaim['scores'][b_id]
        row.update(bertscore_gold_tokens=sa['gold_tokens'],bertscore_a_tokens=sa['candidate_tokens'],bertscore_b_tokens=sb['candidate_tokens'],
                   bertscore_gold_truncated=sa['gold_truncated'],bertscore_a_truncated=sa['candidate_truncated'],bertscore_b_truncated=sb['candidate_truncated'],
                   bertscore_truncated=bool(sa['gold_truncated'] or sa['candidate_truncated'] or sb['candidate_truncated']),
                   patclaim_a_tokens=pa['pair_tokens'],patclaim_b_tokens=pb['pair_tokens'],
                   patclaim_a_truncated=pa['truncated'],patclaim_b_truncated=pb['truncated'],
                   patclaim_truncated=bool(pa['truncated'] or pb['truncated']))
        for method in METHODS:
            va, vb = (sa[method], sb[method]) if method in PRIMARY[:4] else (pa[method],pb[method])
            va, vb = float(va), float(vb)
            if not all(math.isfinite(v) for v in [va, vb]):
                raise ValueError(f'Nonfinite score {row["record_id"]}:{method}')
            prediction = preference(va-vb)
            row['methods'][method] = dict(score_a=va,score_b=vb,delta=va-vb,prediction=prediction,correct=prediction==row['human_label'])
        result.append(row)
    return result


def write_csvs(out, rows):
    base = ['record_id','split','row_number','gold_id','human_label','gold_claim','A','B','pair_a_id','pair_b_id',
            'duplicate_full_row','overlap_train_comparison','contains_gold_copy','bertscore_truncated','bertscore_gold_tokens',
            'bertscore_a_tokens','bertscore_b_tokens','bertscore_gold_truncated','bertscore_a_truncated','bertscore_b_truncated',
            'patclaim_truncated','patclaim_a_tokens','patclaim_b_tokens','patclaim_a_truncated','patclaim_b_truncated']
    fields = base + [f'{method}_{k}' for method in METHODS for k in ['score_a','score_b','delta','prediction','correct']]
    with (out / 'scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {k:row[k] for k in base}
            item.update({f'{method}_{k}':v for method in METHODS for k,v in row['methods'][method].items()})
            writer.writerow(item)
    fields = ['record_id','split','row_number','gold_id','human_label','method','score_a','score_b','delta','prediction','correct']
    with (out / 'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                writer.writerow(dict({k:row[k] for k in fields[:5]},method=method,**row['methods'][method]))


def select_cases(test):
    selected=[]
    selected_ids=set()
    def add(candidates, reason):
        for row in candidates:
            if row['record_id'] not in selected_ids:
                item={k:row[k] for k in ['record_id','split','row_number','gold_claim','A','B','human_label','methods','bertscore_truncated','patclaim_truncated']}
                item['reason_selected']=reason
                selected.append(item)
                selected_ids.add(row['record_id'])
                break
    add(sorted([r for r in test if not r['methods']['patclaimeval']['correct']],key=lambda r:-abs(r['methods']['patclaimeval']['delta'])), 'PatClaimEval 分數差較大但與專家答案不一致')
    add(sorted([r for r in test if r['methods']['patclaimeval']['correct']],key=lambda r:(sum(r['methods'][m]['correct'] for m in PRIMARY[:4]),r['row_number'])), 'PatClaimEval 判斷正確而其他指標較多判錯')
    add(sorted([r for r in test if not r['methods']['bertscore_f1']['correct']],key=lambda r:-min(r['methods']['bertscore_f1']['score_a'],r['methods']['bertscore_f1']['score_b'])), '兩份候選的 BERTScore 均高但偏好判斷仍錯誤')
    add([r for r in test if r['human_label']==0], '專家認為平手的不同文字候選')
    add([r for r in test if r['patclaim_truncated']], 'PatClaimEval 2048 tokens 截斷的長文本')
    add([r for r in test if r['overlap_train_comparison']], '與訓練集重複的測試比較')
    add([r for r in test if r['contains_gold_copy']], '含與參考原文完全相同的候選')
    add(sorted(test,key=lambda r:(-sum(r['methods'][m]['correct'] for m in PRIMARY),r['row_number'])), '五項指標判斷較一致的對照案例')
    if len(selected)!=8:
        for row in test:
            add([row], '補充比較案例')
            if len(selected)==8:
                break
    return selected


def aggregate(out, replicates=10000):
    out=Path(out)
    prepared=load_json(out/'pairs.json')
    similarity=load_json(out/'similarity_scores.json')
    patclaim=load_json(out/'patclaim_scores.json')
    input_hash=digest(out/'pairs.json')
    for stage in [similarity,patclaim]:
        if stage['status']!='complete' or stage['input_sha256']!=input_hash:
            raise ValueError('Incomplete/mismatching worker result')
    bert_revision='722cf37b1afa9454edce342e7895e588b6ff1d59'
    bert_snapshot=PROJECT/'LLM4DPCG'/'.cache'/'huggingface'/'hub'/'models--roberta-large'/'snapshots'/bert_revision
    model_provenance=dict(
        bertscore=dict(repository='roberta-large',revision=bert_revision,
                       files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(bert_snapshot.iterdir()) if p.is_file()}),
        patclaimeval=patclaim['model'],
    )
    atomic_json(out/'model_provenance.json',model_provenance)
    rows=join_rows(prepared,similarity,patclaim)
    cohorts={
        'test':('test','官方測試集 全部原始比較', [r for r in rows if r['split']=='test']),
        'train':('train','官方訓練集 僅為訓練資料上的描述結果',[r for r in rows if r['split']=='train']),
        'test_no_train_comparison':('test','排除與訓練集重複的9筆比較',[r for r in rows if r['split']=='test' and not r['overlap_train_comparison']]),
        'test_no_gold_copy':('test','排除含原文gold候選的28筆比較',[r for r in rows if r['split']=='test' and not r['contains_gold_copy']]),
        'train_deduplicated':('train','訓練集去除20筆完全重複紀錄',[r for r in rows if r['split']=='train' and not r['duplicate_full_row']]),
        'test_bertscore_truncated':('test','BERTScore 至少一段文字超過512 tokens',[r for r in rows if r['split']=='test' and r['bertscore_truncated']]),
        'test_bertscore_untruncated':('test','BERTScore 所有文字皆未超過512 tokens',[r for r in rows if r['split']=='test' and not r['bertscore_truncated']]),
        'train_bertscore_truncated':('train','訓練集BERTScore超長群組',[r for r in rows if r['split']=='train' and r['bertscore_truncated']]),
        'train_bertscore_untruncated':('train','訓練集BERTScore未超長群組',[r for r in rows if r['split']=='train' and not r['bertscore_truncated']]),
    }
    results, cohort_results, paired = {},{},[]
    for key,(split,description,items) in cohorts.items():
        print(f'Aggregating {key}: {len(items)} rows',flush=True)
        statistics={method:method_statistics(items,method) for method in METHODS}
        cis,differences=cluster_bootstrap(items,replicates)
        for method in METHODS:
            statistics[method]['ci']=cis[method]
        for item in differences:
            a,b=item['method_a'],item['method_b']
            item.update(cohort=key,delta_accuracy=statistics[a]['accuracy']-statistics[b]['accuracy'],delta_macro_f1=statistics[a]['macro_f1']-statistics[b]['macro_f1'])
            paired.append(item)
        cohort_results[key]=dict(split=split,description=description,n=len(items),results=statistics)
        if key in ['train','test']:
            results[key]=statistics
    train=cohorts['train'][2]
    test=cohorts['test'][2]
    majority=Counter(r['human_label'] for r in train).most_common(1)[0][0]
    baselines=dict(majority=dict(label=majority,train=classification([r['human_label'] for r in train],[majority]*len(train)),test=classification([r['human_label'] for r in test],[majority]*len(test))),uniform_random_expected_accuracy=1/3)
    audit=prepared['audit']
    audit['truncation']={}
    for split,items in [('train',train),('test',test)]:
        audit['truncation'][split]=dict(
            bertscore_rows=sum(r['bertscore_truncated'] for r in items),
            bertscore_gold=sum(r['bertscore_gold_truncated'] for r in items),
            bertscore_a=sum(r['bertscore_a_truncated'] for r in items),
            bertscore_b=sum(r['bertscore_b_truncated'] for r in items),
            bertscore_pair_instances=sum((r['bertscore_gold_truncated'] or r['bertscore_a_truncated'])+(r['bertscore_gold_truncated'] or r['bertscore_b_truncated']) for r in items),
            patclaim_rows=sum(r['patclaim_truncated'] for r in items),
            patclaim_pair_instances=sum(r['patclaim_a_truncated']+r['patclaim_b_truncated'] for r in items),
            patclaim_affected_row_numbers=[r['row_number'] for r in items if r['patclaim_truncated']])
    paper=[]
    for method,(accuracy,f1,tau,rho) in PAPER.items():
        current=results['test'][method]
        paper.append(dict(method=method,accuracy=accuracy,reported_f1=f1,kendall_tau=tau,spearman_rho=rho,
                          our_accuracy_gap_pp=100*(current['accuracy']-accuracy),our_weighted_f1_gap_pp=100*(current['weighted_f1']-f1),
                          our_macro_f1_gap_pp=100*(current['macro_f1']-f1)))
    cases=select_cases(test)
    observation_path=out/'case_observations.json'
    if observation_path.exists():
        observations=load_json(observation_path)
        for case in cases:
            observation=observations.get(case['record_id'])
            if observation:
                for evidence in observation.get('evidence',[]):
                    if evidence['field'] not in ['gold_claim','A','B'] or not evidence['quote'] or evidence['quote'] not in case[evidence['field']]:
                        raise ValueError('Case textual evidence is not an exact source substring')
                case['text_observation']=observation['observation']
                case['evidence']=observation.get('evidence',[])
                case['interpretation_limit']=observation.get('interpretation_limit','僅為原文觀察，不代表專家的原始判斷理由')
    write_csvs(out,rows)
    atomic_json(out/'data_dictionary.json',dict(
        encoding='CSV 為 UTF-8 BOM，標準引號及多行欄位；勿逐文字行計算紀錄筆數',
        scores_csv_rows=1228,predictions_csv_rows=7368,
        base_fields={
            'record_id':'固定切分及一基列號，例如 test_0026',
            'split':'train 或 test，統計分開呈現',
            'row_number':'在原始 JSON 陣列的一基列號',
            'gold_id':'原始 gold_claim UTF-8 字串 SHA-256，作為分群重抽樣單位',
            'human_label':'專家答案：1=A較佳，0=平手，-1=B較佳',
            'gold_claim':'未自行正規化的完整參考文字',
            'A':'未自行正規化的完整候選A',
            'B':'未自行正規化的完整候選B',
            'pair_a_id':'原始 gold/A 配對識別碼',
            'pair_b_id':'原始 gold/B 配對識別碼',
            'duplicate_full_row':'此列是否為同一切分中完全相同紀錄的第二次或之後出現',
            'overlap_train_comparison':'測試列是否與訓練比較重複，A/B交換亦納入檢查',
            'contains_gold_copy':'任一候選是否逐字等於 gold_claim',
            'bertscore_truncated':'gold/A/B 任一文本超過標準512 tokens',
            'patclaim_truncated':'任一 gold/candidate 配對超過官方2048 tokens',
        },
        token_field_patterns={
            'bertscore_*_tokens':'包含特殊token的未截斷單一文本token數',
            'bertscore_*_truncated':'對應文本是否超過512 tokens',
            'patclaim_*_tokens':'包含特殊token的未截斷 gold/candidate 合併配對token數',
            'patclaim_*_truncated':'對應配對是否超過2048 tokens',
        },
        methods=LABELS,primary_methods=PRIMARY,
        method_column_suffixes={
            'score_a':'相同方法下的 score(gold,A)，不是正確機率',
            'score_b':'相同方法下的 score(gold,B)，不是正確機率',
            'delta':'未四捨五入的 score_a - score_b',
            'prediction':'abs(delta)<0.0001為0，否則依正負為1或-1',
            'correct':'prediction 是否等於 human_label',
        },
        confusion_matrix='固定標籤順序[-1,0,1]；列為專家答案，欄為模型預測',
        ci='依gold文字群組重抽樣的百分位95%信賴區間，非獨立逐列抽樣',
    ))
    atomic_json(out/'comparisons.json',rows)
    result=dict(schema_version=1,metadata=dict(created_at_taipei=stamp(),seed=42,bootstrap_replicates=replicates,
                  source_hashes=prepared['source_hashes'],input_sha256=input_hash,
                  source_manifest=prepared['source_manifest'],python=sys.version,platform=platform.platform(),
                  case_observations_sha256=digest(observation_path) if observation_path.exists() else None,
                  program_hashes={p.name:digest(p) for p in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent.parent/'run_patent_ce_evaluation.py']},
                  model_provenance=model_provenance,
                  worker_output_hashes={name:digest(out/name) for name in ['similarity_scores.json','patclaim_scores.json']},
                  epsilon=EPSILON,tie_policy='abs(score_a-score_b) < 0.0001; equality at epsilon is not a tie',
                  bootstrap_policy='Gold-reference cluster resampling with replacement; g draws from the g gold groups present in each cohort (base train/test each20); paired draws shared by all methods; percentile95% CI; all original row weights retained'),
                audit=audit,method_labels=LABELS,methods=METHODS,primary_methods=PRIMARY,
                settings=dict(similarity=similarity['settings'],patclaim=patclaim['settings']),
                environments=dict(similarity=similarity['environment'],patclaim=patclaim['environment']),
                results=results,cohorts=cohort_results,paired_differences=paired,baselines=baselines,
                paper_comparison=paper,cases=cases,
                verification=dict(status='pending_independent_csv_verification',source_rows=len(rows),unique_pairs=len(prepared['pairs']),
                                  all_pairs_scored=True,raw_text_retained=True,skipped_rows=0))
    atomic_json(out/'results.json',result)
    fields=['cohort','method','n','correct','accuracy','error_rate','macro_f1','weighted_f1','tie_recall','non_tie_accuracy','reverse_rate','continuous_tau','continuous_rho','discrete_tau','discrete_rho','accuracy_ci_low','accuracy_ci_high','macro_f1_ci_low','macro_f1_ci_high']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for cohort,value in cohort_results.items():
            for method,s in value['results'].items():
                item=dict(cohort=cohort,method=method,**{k:s[k] for k in fields[2:-4]})
                item.update(accuracy_ci_low=s['ci']['accuracy'][0],accuracy_ci_high=s['ci']['accuracy'][1],macro_f1_ci_low=s['ci']['macro_f1'][0],macro_f1_ci_high=s['ci']['macro_f1'][1])
                writer.writerow(item)
    print('Aggregate complete: '+str(out/'results.json'),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap-replicates',type=int,default=10000)
    args=parser.parse_args()
    if args.bootstrap_replicates<100:
        parser.error('Use at least100 bootstrap replicates')
    aggregate(args.output_dir,args.bootstrap_replicates)
