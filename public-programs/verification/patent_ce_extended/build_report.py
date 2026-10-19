"""Build the complete Traditional Chinese thirteen-method report from verified results."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx.shared import Pt

from core import PROJECT, PRIMARY, METHODS, NEW_METHODS, LABELS, atomic_json, digest, load_json, stamp

spec = importlib.util.spec_from_file_location('patent_ce_legacy_report', Path(__file__).parents[1]/'patent_ce/build_report.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
NAME = 'Patent_CE_十三項方法評估與專家答案比較報告'
legacy.REPORT_NAME = NAME
pct, num, interval = legacy.pct, legacy.num, legacy.interval
CHOICES = {-1:'B 較佳',0:'平手',1:'A 較佳'}

SETTINGS = [
 ('BLEU-1／BLEU-4','Tokenizer13a；max_order=1／4；smooth=False；逐候選。'),
 ('ROUGE-1／2／L F1','逐候選 F1；不使用詞幹化，非 corpus 平均回填。'),
 ('METEOR','NLTK tokenize／WordNet；α=0.9、β=3、γ=0.5；逐配對計算。'),
 ('BERTScore F1','roberta-large 第17層；IDF 關閉；baseline rescaling 關閉；slow tokenizer；512 tokens。'),
 ('BARTScore CNN','facebook/bart-large-cnn；gold→candidate；來源及目標各1,024 tokens；平均目標token log probability，通常為負數，越高越好。'),
 ('MoverScore v2','distilbert-base-uncased；512 tokens；最後一層；unigram；移除 subwords／標點；stopwords=[]。'),
 ('SimCSE','sup-simcse-bert-base-uncased；supervised CLS pooler；128 tokens；配對 cosine。'),
 ('UniEval relevance','MingZhong/unieval-sum；只計 relevance；output=candidate、reference=gold、source 空字串；合併輸入1,024。'),
 ('AlignScore large','large checkpoint／roberta-large；gold=context、candidate=claim；nli_sp；官方分句與分塊聚合。'),
 ('PatClaimEval','固定 Quality checkpoint；官方2,048 tokens；4,096 tokens 為附加敏感度設定。'),
]

def ci_of(stat, key='accuracy'):
    return stat.get('ci',{}).get(key,[stat[key],stat[key]])

def f1_interval(bounds):
    return '不適用' if bounds is None else f'[{num(bounds[0])}, {num(bounds[1])}]'

def point_number(value):
    text=f'{100*value:+.2f}'
    return '0.00' if text in ['+0.00','-0.00'] else text

def points_interval(bounds):
    return '不適用' if bounds is None else f'[{point_number(bounds[0])}, {point_number(bounds[1])}]'

def mark_artifact(out, kind):
    """Run the bundled authoring marker once per deliverable in this run."""
    path=out/'artifact_operation_markers.json'
    record=load_json(path) if path.exists() else {}
    if kind not in record:
        node=Path(r'C:\Users\agent\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        versions={'docx':('documents','26.1011.51657'),'pdf':('pdf','26.1011.51657')}
        package,version=versions[kind]
        marker=Path(r'C:\Users\agent\.codex\plugins\cache\openai-primary-runtime')/package/version/'skills'/package/'container_tools/mark_artifact_operation_started.mjs'
        subprocess.run([str(node),str(marker),'--operation-kind','create','--expected-output-count','1','--output-format',kind],check=True)
        record[kind]=dict(marked_at=stamp(),operation='create',expected_output_count=1)
        atomic_json(path,record)

def make_figures(result, out):
    legacy.setup_plots()
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    figures = {}
    test = result['results']['test']
    order = sorted(PRIMARY,key=lambda m:test[m]['accuracy'],reverse=True)
    fig,ax=plt.subplots(figsize=(9.2,6.3),constrained_layout=True)
    values=np.array([test[m]['accuracy'] for m in order])*100
    ci=np.array([ci_of(test[m]) for m in order])*100
    pos=np.arange(len(order))
    ax.barh(pos,values,color=['#345E7C' if m!='patclaimeval' else '#487D65' for m in order],height=.64)
    ax.errorbar(values,pos,xerr=np.maximum(0,np.array([values-ci[:,0],ci[:,1]-values])),fmt='none',ecolor='#222222',capsize=3)
    for i,value in enumerate(values):
        ax.text(max(value,ci[i,1])+1.2,i,f'{value:.2f}%',va='center',fontsize=11.5)
    baseline=result['baselines']['majority']['test']['accuracy']*100
    ax.axvline(baseline,color='#9B6848',ls='--',label=f'多數類基準 {baseline:.2f}%')
    ax.set_yticks(pos,[LABELS[m] for m in order],fontsize=11.5);ax.invert_yaxis()
    ax.set_xlim(0,max(82,ci[:,1].max()+12));ax.set_xlabel('測試集偏好判斷正確率  %',fontsize=12)
    ax.tick_params(axis='x',labelsize=11.5)
    ax.legend(loc='lower right',fontsize=11.5);ax.grid(axis='x',alpha=.15)
    figures['accuracy']=folder/'test_accuracy_ci.png';fig.savefig(figures['accuracy']);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.3,6.4),constrained_layout=True,sharey=True)
    for ax,key,title in zip(axes,['continuous_rho','discrete_rho'],['連續分數差與專家答案','離散預測與專家答案']):
        vals=[test[m].get(key) for m in PRIMARY]
        ax.barh(range(len(PRIMARY)),[v if v is not None else 0 for v in vals],height=.6,color='#496D88')
        for i,v in enumerate(vals):
            ax.text((v or 0)+.01,i,num(v,3),va='center',fontsize=12.5)
        ax.set_title(title,fontsize=13);ax.set_xlabel('Spearman ρ',fontsize=13);ax.set_xlim(min(-.1,min(v or 0 for v in vals)-.05),max(.8,max(v or 0 for v in vals)+.15));ax.grid(axis='x',alpha=.15)
        ax.tick_params(axis='x',labelsize=12.5)
    axes[0].set_yticks(range(len(PRIMARY)),[LABELS[m] for m in PRIMARY],fontsize=13);axes[0].invert_yaxis()
    figures['correlations']=folder/'test_spearman_two_definitions.png';fig.savefig(figures['correlations']);plt.close(fig)
    for group,methods in enumerate([PRIMARY[:5],PRIMARY[5:9],PRIMARY[9:]],1):
        fig,axes=plt.subplots(2,3,figsize=(9.4,6.4),constrained_layout=True)
        for i,method in enumerate(methods):
            ax=axes.flat[i];matrix=np.array(test[method]['confusion_matrix'])
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(matrix.max(),1))
            for y in range(3):
                for x in range(3):
                    ax.text(x,y,str(matrix[y,x]),ha='center',va='center',color='white' if matrix[y,x]>matrix.max()*.6 else '#111111',fontsize=12)
            ax.set_xticks([0,1,2],['B 較佳','平手','A 較佳'],fontsize=11.5)
            ax.set_yticks([0,1,2],['B 較佳','平手','A 較佳'],fontsize=11.5)
            ax.set_xlabel('指標預測',fontsize=11.5);ax.set_ylabel('專家答案',fontsize=11.5);ax.set_title(LABELS[method],fontsize=12)
        for ax in list(axes.flat)[len(methods):]:ax.axis('off')
        figures['confusion'+str(group)]=folder/f'test_confusion_group_{group}.png'
        fig.savefig(figures['confusion'+str(group)]);plt.close(fig)
    return figures

def table_main(report, stats, n):
    report.table(['方法','正確筆數','Accuracy','95% 區間','Macro-F1','Weighted-F1'],
        [[LABELS[m],f"{stats[m]['correct']}／{n}",pct(stats[m]['accuracy']),interval(ci_of(stats[m])),num(stats[m]['macro_f1']),num(stats[m]['weighted_f1'])] for m in PRIMARY],
        [3.8,2.0,2.1,4.0,2.5,3.0],font_size=8.8,number_columns=(1,2,3,4,5))

def key_pair(result, a, b, cohort='test', metric='accuracy'):
    for item in result.get('paired_differences',[]):
        if item.get('cohort')==cohort and {item['method_a'],item['method_b']}=={a,b}:
            sign=1 if item['method_a']==a else -1
            ci=item['ci_'+metric]
            return [sign*ci[0],sign*ci[1]] if sign==1 else [-ci[1],-ci[0]]
    return None

def build(out):
    result=load_json(out/'results.json')
    verification=load_json(out/'verification.json')
    if verification.get('status') not in ['passed','verified']:
        raise ValueError('Independent verification must pass before report authoring')
    optimization=load_json(out/'alignscore_optimization_validation.json')
    if optimization.get('status')!='passed':
        raise ValueError('Actual AlignScore optimization parity must pass before authoring')
    supplemental=load_json(out/'supplemental_sample_validation.json')
    if supplemental.get('status')!='passed' or supplemental.get('complete') is not True:
        raise ValueError('Shortest/median official sample validation must pass before authoring')
    test,train=result['results']['test'],result['results']['train']
    assert len(PRIMARY)==13 and all(test[m]['n']==184 for m in PRIMARY)
    figures=make_figures(result,out)
    mark_artifact(out,'docx')
    r=legacy.Report(out)
    r.doc.core_properties.title=NAME;r.doc.core_properties.keywords='Patent-CE, thirteen metrics, Quality, expert preference'
    r.doc.add_paragraph('Patent CE 十三項方法評估',style='Title')
    r.doc.add_paragraph('自動指標與專家品質偏好一致程度的繁體中文研究報告',style='Subtitle')
    r.p('評估範圍  全部1,228筆比較  訓練1,044筆  測試184筆',bold=True)
    r.p('執行識別  '+out.name+'  |  主要方法13項  |  G-Eval-4未執行',size=9.2)
    r.heading('研究摘要')
    best=max(PRIMARY,key=lambda m:test[m]['accuracy'])
    best_methods=[m for m in PRIMARY if test[m]['correct']==test[best]['correct']]
    best_text='、'.join(LABELS[m] for m in best_methods)+('（並列）' if len(best_methods)>1 else '')
    r.p(f"本報告重新評估公開Patent-CE資料，檢驗十三種自動方法對候選A、B的品質排序是否符合專家Quality答案。測試集正確率點估計最高者為{best_text}，正確{test[best]['correct']}／184筆，Accuracy {pct(test[best]['accuracy'])}。各方法的Macro-F1與不確定性區間詳見主要結果表。")
    if len(best_methods)==1:
        runner=max([m for m in PRIMARY if m!=best],key=lambda m:test[m]['accuracy'])
        gap=test[best]['accuracy']-test[runner]['accuracy']
        r.p(f"{LABELS[best]}的Accuracy 95%信賴區間為{interval(ci_of(test[best]))}，Macro-F1為{num(test[best]['macro_f1'])}。相較次高的{LABELS[runner]}，Accuracy差距為{point_number(gap)}個百分點，配對95%差距區間{points_interval(key_pair(result,best,runner))}個百分點。")
        all_cis=[key_pair(result,best,m,metric=metric) for m in PRIMARY if m!=best for metric in ['accuracy','macro_f1']]
        if all(c is not None and c[0]<=0<=c[1] for c in all_cis):
            r.p('最高點估計方法與其餘十二項方法的Accuracy及Macro-F1配對差距95%區間均包含0。依本次資料，尚不能確認其有明確優勢。')
    r.p('正確率是三分類偏好一致率；分數本身不是專利正確機率。方法間差異需結合按參考專利分群的信賴區間解讀，不能只以名次認定某方法穩定較佳。')
    r.p('公開檔案僅有Quality逐筆專家標籤。G-Eval-4依本次範圍未使用付費API；Completeness、Clarity、Consistency、Linkage不產生自行推測的成效。各新增方法採明確宣告設定，與原論文未公開的細節不保證一致。')
    r.p('測試資料的全部參考與候選配對均曾出現在訓練集。本文結論限定官方切分下對已見候選的偏好一致程度，不作未見專利泛化推論。')
    r.heading('資料與專家答案',new_page=True)
    r.p('每筆包含參考請求項gold_claim、候選A、候選B與human_eval.Quality。標籤1表示A較佳、0表示平手、-1表示B較佳。所有原始字串逐位元保留，不自行整理空白或改寫換行；各方法自行分詞的行為另行記錄。')
    prepared=load_json(out/'pairs.json')
    r.table(['資料切分','比較數','參考組數','A較佳','平手','B較佳'],[[s,prepared['audit'][s]['rows'],prepared['audit'][s]['unique_gold'],prepared['audit'][s]['label_counts']['1'],prepared['audit'][s]['label_counts']['0'],prepared['audit'][s]['label_counts']['-1']] for s in ['train','test']],[3.4,2.8,2.8,2.8,2.8,2.8],number_columns=(1,2,3,4,5))
    r.p('1,228筆比較包含224組不同的「原始參考文字、原始候選文字」配對。每方法先評分唯一配對，再回填各列；去重只節省推論，主要統計仍依原始列數加權。')
    r.table(['稽核項目','結果'],[['測試參考是否見於訓練','20組全部曾出現'],['測試不同參考／候選配對','186組全部曾出現'],['測試與訓練重複比較','9筆，A／B交換亦納入檢查'],['測試含原文gold候選','28筆'],['訓練完全重複列','20筆，補充去重後1,024筆結果']],[6.0,11.4])
    r.heading('方法設定與分數方向',new_page=True)
    r.table(['方法','固定設定'],[list(x) for x in SETTINGS],[4.1,13.3],font_size=9.0)
    r.p('十三項方法均以分數較高為較佳；各方法分數都不是答案正確的機率。')
    r.p('BARTScore是平均目標token log probability，數值通常為負，較接近零較佳，不能取絕對值後判勝負。SimCSE只取一對文字的cosine，不把所有配對形成的矩陣平均。UniEval只使用relevance，不能將其原生構面名稱當作專利四個構面的人工答案。')
    r.p('原文保留是指不在方法外先改寫資料。Tokenizer13a本身會處理HTML實體、換行與部分標點，並按空白分詞，保留大小寫；ROUGE預設tokenizer會轉小寫，以空白替代非a–z／0–9字元，且本次不做詞幹化。METEOR使用NLTK分詞，內部採預設小寫、Porter詞幹及WordNet同義詞匹配。神經方法使用各自固定的tokenizer；DistilBERT與本次SimCSE為uncased設定。這些方法內部處理和特殊token均不會回寫覆蓋CSV中的來源原文。',size=9.3)
    r.heading('各方法分數的含義',new_page=True)
    r.p('BLEU-1衡量候選unigram的裁切匹配精確率，再乘上過短候選的長度懲罰。BLEU-4將1至4-gram精確率作等權幾何平均，再乘同一懲罰；本次不平滑，任一必要階的精確率為零時，整體分數即為零。它們強調局部詞序與字面重疊。')
    r.p('ROUGE-1與ROUGE-2分別依unigram與bigram重疊計算precision、recall，再取F1。ROUGE-L以最長共同子序列的長度計算兩者，容許中間插入不同詞語，但仍依賴詞彙與相對順序。')
    r.p('METEOR依序匹配原詞、詞幹及WordNet同義詞，再計算偏重recall的加權調和平均，並以匹配片段的破碎程度扣分。因此即使重疊詞數相近，排列與同義詞匹配仍可能改變分數。')
    r.p('BERTScore先為參考與候選的各token建立上下文向量，再以最佳token匹配的相似度形成precision與recall，最後取F1。它的F1是在文本配對內聚合向量相似度，與本報告跨184筆偏好分類計算的Macro-F1不同。')
    r.p('BARTScore是把gold當輸入時，模型對候選各目標token的條件對數機率取平均；越容易在該模型下生成，分數越高。這是生成模型的條件分數，不能當作專家品質正確率。')
    r.p('MoverScore v2為上下文token向量與訓練語料IDF建立加權分布，以Earth Mover’s Distance求搬運成本，將成本轉成越高越好的相似度。參考與候選使用各自IDF，故即使原文複製也不保證剛好得到1。SimCSE則將兩段文本各壓成supervised CLS句向量，再取配對cosine；本次只使用各段開頭最多128 tokens。')
    r.p('UniEval relevance透過固定問題模板，取模型回答Yes相對於Yes與No的權重作為相關程度。這個模型內部的二選一歸一化分數，也不是對Patent-CE專家答案校準後的機率。')
    r.p('AlignScore nli_sp將每個候選句子與參考片段配對，先取該候選句子在參考片段上的最佳對齊，再平均各候選句子的分數。它評估句子被參考內容支持的程度；分句、分塊與每次模型輸入的截斷共同決定實際可見內容。')
    r.p('PatClaimEval使用固定Quality權重對參考與候選產生品質分數。本報告把同一參考下A、B的分數差轉成偏好，再與公開專家標籤核對；沒有逐候選的絕對人工分數可用來計算絕對品質誤差。')
    r.heading('實作相容性與可重現性',new_page=True)
    r.p('既有五項主要方法與PatClaimEval長度變體從前次固定產物匯入，不重新推論。來源交付清單、資料、設定及本機模型檔案均經SHA-256核對，所有候選分數未四捨五入。新方法在隔離WSL環境執行，保留舊程式及環境。')
    threads='/'.join(str(v) for v in sorted({result['settings'][m]['threads'] for m in NEW_METHODS}))
    batches='/'.join(str(v) for v in sorted({result['settings'][m]['batch_size'] for m in NEW_METHODS}))
    r.p(f'新增神經方法使用Python 3.10.20、PyTorch 1.13.1 CPU、Transformers 4.26.1、NumPy 1.23.5及Lightning 1.9.5；CPU float32、{threads} threads、batch size {batches}、seed 42。固定官方原始碼與模型revision，完整程式差異、模型檔案hash與套件鎖定檔隨附件保存。')
    r.p('適配器處理CPU裝置、模型載入及套件介面相容；AlignScore另快取完全相同的參考片段／候選句子之三個原生輸出head，保留原有分塊、分詞與max／mean聚合。小樣本比對明確繞過快取執行官方函式，確認數值在10⁻⁵誤差內一致。')
    short,middle=supplemental['selection']['samples']
    r.p(f"初始三組smoke包含gold-copy與最長配對。稽核後另於正式推論完成時補驗真正最短的{short['combined_characters']:,}字元配對，以及最接近全體中位數的{middle['combined_characters']:,}字元配對；長度為原始參考加候選的Unicode字元總數。八個新增方法均再次對官方函式與正式分數核對，未改動評分設定、模型或門檻。補驗使用獨立輸入及checkpoint，MoverScore仍保留全部訓練原文建立IDF。此補驗時序與原定事前涵蓋短、中、長樣本的安排不同，完整紀錄見supplemental_sample_validation.json。")
    proof_text='逐張量完全相等' if optimization.get('all_head_tensors_exactly_equal') else '逐head誤差不超過10⁻⁷'
    r.p('AlignScore評分也略去未被三個評分head使用的MLM詞彙logits；完整模型模組與checkpoint參數仍保留，僅在eval且沒有MLM標籤時省算。對同一輸入執行原始與優化forward，三個head'+proof_text+'；最終smoke另以無快取、未省算的原始函式核對整體分數。證據見alignscore_optimization_validation.json。')
    r.p('MoverScore明確使用官方v2 DistilBERT版本，不視為原始BERT-MNLI版本的完全重現。參考IDF與候選IDF分別由訓練集去重原文建立，固定保存；不使用測試人工答案建立權重或調整設定。')
    r.p('所有指標使用原始分數尺度及相同平手門檻；尺度差異會影響平手頻率，因此本文同時列出平手召回率、非平手正確率及分數差分布。沒有以測試資料校準分數或挑選門檻。')
    r.heading('執行時間與小樣本資源',new_page=True)
    resources=[]
    for m in NEW_METHODS:
        formal=load_json(out/(m+'_scores.json'))
        measured=supplemental['methods'][m]['resources']
        resources.append([LABELS[m],f"{formal['elapsed_seconds']/60:.2f}",f"{measured['initial_wall_seconds']/60:.2f}",f"{measured['initial_peak_rss_kbytes']/1024**2:.3f}"])
    r.table(['新增方法','正式224對時間 分','補驗2對程序時間 分','補驗首次程序峰值 GiB'],resources,[5.4,4.0,4.0,4.0],font_size=8.8,number_columns=(1,2,3))
    r.p('正式時間為worker保存的首次elapsed_seconds，保留零新增推論續跑前的數值。補驗時間與記憶體由GNU /usr/bin/time -v實測整個獨立worker子程序，包含載入、兩組樣本、官方函式核對及續跑驗證；不是正式224對的峰值，也不是模型參數本身的大小。GiB按GNU time kbytes除以1,048,576換算。')
    r.p('補驗資源量測於完整正式推論後進行；首次程序與後續純續跑的紀錄分開保留，避免低記憶體的續跑程序覆蓋原始峰值。原始量測文字及SHA-256隨附件保存。上述時間只描述本機此次執行，不作跨硬體速度結論。')
    r.heading('偏好判斷與統計公式')
    r.p('令 Δ = score(gold,A) − score(gold,B)。僅當 |Δ| < 0.0001 時判為平手；等於門檻仍依正負判勝負。Accuracy = 正確筆數／比較筆數。各類F1 = 2TP／(2TP+FP+FN)；Macro-F1為三類等權平均，Weighted-F1按專家類別支持筆數加權。分母為零時該類指標設為0。')
    r.p('Kendall τ-b納入平手校正，Spearman ρ為秩相關。連續版本以Δ對專家標籤計算；離散版本以三分類預測對專家標籤計算。常數輸出無法定義相關係數時標為不適用。')
    r.heading('測試集主要結果',new_page=True)
    table_main(r,test,184)
    r.p('上表為全部184筆官方測試比較，未排除困難、超長或重複資料。Macro-F1與Weighted-F1是偏好分類成效，和ROUGE／BERTScore名稱中的文本相似度F1不同。')
    majority=result['baselines']['majority']
    r.p(f"多數類基準只依訓練集決定，固定預測「{CHOICES[majority['label']]}」。測試集正確{majority['test']['correct']}／184筆，Accuracy {pct(majority['test']['accuracy'])}，Macro-F1 {num(majority['test']['macro_f1'])}，Weighted-F1 {num(majority['test']['weighted_f1'])}；未用測試答案選擇基準類別。")
    r.heading('正確率與分群不確定性',new_page=True)
    r.figure(figures['accuracy'],'圖一  十三項方法測試正確率與95%分群bootstrap信賴區間')
    r.p(f"信賴區間按完整gold文字形成參考專利群組，每次從g個群組有放回抽g次，保留群組內原始列權重。共{result['metadata']['bootstrap_replicates']:,}次，seed {result['metadata']['seed']}；同一抽樣用於所有方法，故能估計成對差異。群組數僅20，區間仍可能較寬。")
    pairedrows=[]
    for method in PRIMARY:
        if method=='patclaimeval':continue
        gap=test['patclaimeval']['accuracy']-test[method]['accuracy']
        c=key_pair(result,'patclaimeval',method)
        pairedrows.append([LABELS[method],point_number(gap),points_interval(c), '未計算' if c is None else ('包含0' if c[0]<=0<=c[1] else '未包含0')])
    r.table(['PatClaimEval 對照方法','差距百分點','差距95%區間 百分點','區間是否含0'],pairedrows,[5.1,3.0,5.7,3.6],font_size=8.7,number_columns=(1,2,3))
    r.p('此處區間為探索性成對比較，未作多重比較校正；只有點估計較高不足以認定穩定優勢。全部78組主要方法配對差異詳見JSON與附件。')
    r.heading('與專家答案的相關係數',new_page=True)
    r.table(['方法','連續 τ-b','連續 ρ','離散 τ-b','離散 ρ'],[[LABELS[m],num(test[m]['continuous_tau']),num(test[m]['continuous_rho']),num(test[m]['discrete_tau']),num(test[m]['discrete_rho'])] for m in PRIMARY],[4.6,3.2,3.2,3.2,3.2],number_columns=(1,2,3,4))
    r.p('連續相關反映分數差與專家偏好之間的排序關係；離散相關反映經平手門檻轉換後的預測排序。兩者資料輸入不同，不應互換，也不能直接解讀成正確率。')
    r.figure(figures['correlations'],'圖二  連續與離散兩種定義下的Spearman相關',width=6.65)
    r.heading('訓練集描述結果',new_page=True)
    table_main(r,train,1044)
    r.p('訓練集成效為描述性結果。PatClaimEval權重曾以相關資料訓練，且官方切分有內容重疊；訓練成效不能當作獨立驗證。主要比較仍以測試集為準。')
    r.heading('Macro F1分群信賴區間',new_page=True)
    r.table(['方法','測試Macro-F1','測試95%區間','訓練Macro-F1','訓練95%區間'],[[LABELS[m],num(test[m]['macro_f1']),f1_interval(ci_of(test[m],'macro_f1')),num(train[m]['macro_f1']),f1_interval(ci_of(train[m],'macro_f1'))] for m in PRIMARY],[4.2,2.4,4.2,2.4,4.2],font_size=8.8,number_columns=(1,2,3,4))
    r.p('Macro-F1區間採與Accuracy相同的參考專利分群抽樣。全部方法間的Accuracy及Macro-F1配對差距區間保存在results.json。')
    r.heading('平手與方向性錯誤',new_page=True)
    r.table(['方法','錯誤率','平手召回','非平手正確率','完全判反率'],[[LABELS[m],pct(test[m]['error_rate']),pct(test[m]['tie_recall']),pct(test[m]['non_tie_accuracy']),pct(test[m]['reverse_rate'])] for m in PRIMARY],[4.6,3.2,3.2,3.2,3.2],number_columns=(1,2,3,4))
    r.p('平手召回率的分母為專家判平手的49筆；非平手正確率與完全判反率的分母為專家有明確偏好的135筆。完全判反是專家判A較佳卻預測B，或專家判B較佳卻預測A；預測平手屬其他錯誤，不計為完全判反。')
    r.heading('候選分數與分數差分布',new_page=True)
    rows=[]
    for m in PRIMARY:
        d=test[m]['score_statistics']
        rows.append([LABELS[m],num(d['A']['mean']),num(d['B']['mean']),num(d['delta']['q1']),num(d['delta']['median']),num(d['delta']['q3']),test[m]['predicted_tie_n']])
    r.table(['方法','A平均','B平均','Δ第1四分位','Δ中位數','Δ第3四分位','預測平手數'],rows,[4.2,2.2,2.2,2.2,2.2,2.2,2.2],font_size=8.3,number_columns=(1,2,3,4,5,6))
    r.p('分數分布只在相同方法內比較。各方法尺度、訓練目標與截斷規則不同，不能將其平均分數橫向相減，或將數值接近1解讀為內容幾乎完全正確。Δ為未四捨五入的A分數減B分數；本表顯示到四位小數，平手判斷仍使用完整精度。')
    for group,methods in enumerate([PRIMARY[:5],PRIMARY[5:9],PRIMARY[9:]],1):
        r.heading(f'混淆矩陣與各類別成效 第{group}組',new_page=True)
        r.figure(figures['confusion'+str(group)],f'圖{group+2}  列為專家答案，欄為指標預測；每個矩陣合計184筆')
        rows=[]
        for m in methods:
            for k in [-1,0,1]:
                s=test[m]['classes'][str(k)]
                rows.append([LABELS[m],CHOICES[k],num(s['precision']),num(s['recall']),num(s['f1']),s['support']])
        r.table(['方法','專家類別','Precision','Recall','F1','支持筆數'],rows,[4.1,2.6,2.8,2.8,2.8,2.3],font_size=8.3,number_columns=(2,3,4,5))
    r.heading('資料重複與gold-copy敏感度',new_page=True)
    cohortkeys=['test_no_train_comparison','test_no_gold_copy','train_deduplicated']
    rows=[]
    for m in PRIMARY:
        values=[]
        for key in cohortkeys:
            c=result['cohorts'].get(key)
            values.append(pct(c['results'][m]['accuracy']) if c else '不適用')
        rows.append([LABELS[m],pct(test[m]['accuracy'])]+values)
    r.table(['方法','測試全部','測試去重複','測試去gold','訓練去重'],rows,[4.6,3.2,3.2,3.2,3.2],font_size=8.7,number_columns=(1,2,3,4))
    r.p('測試去重複排除與訓練重複的9筆，剩175筆；測試去gold排除任一候選逐字等於gold的28筆，剩156筆；兩種排除分別計算。訓練去重排除20筆完全重複紀錄，剩1,024筆。這些補充結果不能消除所有訓練／測試內容重疊。')
    r.heading('長文本截斷與分塊',new_page=True)
    rows=[]
    for m in PRIMARY:
        counts=[]
        for split in ['train','test']:
            audit=result['audit']['truncation'][split][m]
            total=sum(audit[k] for k in ['truncated_rows','untruncated_rows','unknown_rows'])
            counts.append(f"{audit['truncated_rows']}／{total}；未知{audit['unknown_rows']}")
        rows.append([LABELS[m]]+counts)
    r.table(['方法','訓練至少一側截斷','測試至少一側截斷'],rows,[6.4,5.5,5.5],number_columns=(1,2))
    r.p('此表依各方法的實際tokenizer及輸入結構判定，不共用BERTScore長度。缺少截斷資訊時保留未知，不歸入未截斷。詞彙方法讀取完整文本；AlignScore先依官方規則分句分塊，再記錄片段輸入是否仍超過模型上限，不能把分塊本身等同丟棄全文。逐配對token與片段資訊在pair_metadata附件。')
    rows=[]
    for m in PRIMARY:
        groups=[]
        for truncated in [True,False]:
            key='test_'+m+('_truncated' if truncated else '_untruncated')
            c=result['cohorts'].get(key)
            groups.append(f"{c['n']}筆  {pct(c['results'][m]['accuracy'])}" if c else '無此群組或不適用')
        rows.append([LABELS[m]]+groups)
    r.table(['方法','受截斷群組正確率','未截斷群組正確率'],rows,[6.4,5.5,5.5],font_size=8.7)
    r.p('受截斷與未截斷群組在內容及難度上也可能不同，兩者差距不能單獨證明截斷造成的因果效果。各群組仍按原始比較列權重計算。')
    r.heading('PatClaimEval長度敏感度',new_page=True)
    r.table(['設定','測試正確率','Macro-F1','Weighted-F1'],[[LABELS[m],pct(test[m]['accuracy']),num(test[m]['macro_f1']),num(test[m]['weighted_f1'])] for m in ['patclaimeval','patclaimeval_4096']],[5.4,4.0,4.0,4.0],number_columns=(1,2,3))
    r.p('2,048與4,096兩個設定均已對全部224組唯一配對評分；本次沿用經雜湊核對的固定結果。只有一組原始分數出現約2.50×10⁻⁶差異，偏好標籤與測試正確率皆不變。4,096為補充設定，不算作第十四種主要方法。')
    r.heading('與原論文公開數值的差距',new_page=True)
    paper=result.get('paper_comparison',[])
    if isinstance(paper,dict):paper=list(paper.values())
    pmap={x['method']:x for x in paper}
    def paper_val(m,*keys):
        x=pmap.get(m,{})
        for key in keys:
            if key in x:return x[key]
        return None
    rows=[]
    for m in PRIMARY:
        a=paper_val(m,'accuracy','paper_accuracy')
        rows.append([LABELS[m],pct(a),pct(test[m]['accuracy']),point_number(test[m]['accuracy']-a) if a is not None else '不適用'])
    rows.append(['G-Eval-4 僅引用',pct(paper_val('g_eval4','accuracy','paper_accuracy')),'未執行','不適用'])
    r.table(['方法','論文Accuracy','本次Accuracy','差距百分點'],rows,[5.2,4.1,4.1,4.0],number_columns=(1,2,3))
    r.p('原論文Table4呈現相關係數，Table5呈現分類成效。論文未充分指定相關係數使用連續差值或離散標籤、Kendall變體，以及部分模型checkpoint、長度與F1平均方式；本文兩種相關定義分別呈現，避免誤稱精確重現。')
    r.p('差距以論文已顯示的四捨五入數值計算；落在原表顯示精度內的細小差距，可能只是取位差異，不能視為實質成效變化。')
    r.heading('分類F1與相關係數的外部參照',new_page=True)
    f1rows=[[LABELS[m],num(paper_val(m,'reported_f1','f1','paper_f1')),num(test[m]['macro_f1']),num(test[m]['weighted_f1'])] for m in PRIMARY]
    f1rows.append(['G-Eval-4 僅引用',num(paper_val('g_eval4','reported_f1','f1','paper_f1')),'未執行','未執行'])
    r.table(['方法','論文報告F1','本次Macro-F1','本次Weighted-F1'],f1rows,[5.2,4.1,4.1,4.0],number_columns=(1,2,3))
    r.p('論文未明示F1的平均方式，故將其作為外部引用，並同時列出本次兩種固定三分類平均；不事後挑選較接近論文的平均方式作為主要定義。')
    rows=[]
    for m in PRIMARY:
        rows.append([LABELS[m],num(paper_val(m,'kendall_tau','tau','paper_tau')),num(paper_val(m,'spearman_rho','rho','paper_rho')),num(test[m]['continuous_tau']),num(test[m]['continuous_rho']),num(test[m]['discrete_tau']),num(test[m]['discrete_rho'])])
    rows.append(['G-Eval-4 僅引用',num(paper_val('g_eval4','kendall_tau','tau','paper_tau')),num(paper_val('g_eval4','spearman_rho','rho','paper_rho')),'未執行','未執行','未執行','未執行'])
    r.table(['方法','論文 τ','論文 ρ','連續 τ-b','連續 ρ','離散 τ-b','離散 ρ'],rows,[4.2,2.2,2.2,2.2,2.2,2.2,2.2],font_size=8.3,number_columns=(1,2,3,4,5,6))
    r.p('G-Eval-4僅保留論文引用列，實測欄為未執行；不以本機LLM代替後仍使用原方法名稱。新方法的模型、截斷或IDF設定差異，可能影響與論文數值的差距。')
    r.heading('代表性錯誤案例',new_page=True)
    rows=load_json(out/'comparisons.json')
    tests=[x for x in rows if x['split']=='test']
    legacy_obs=load_json(PROJECT/'outputs/patent_ce_evaluation_20261019_203847/case_observations.json')
    chosen=[]
    # Fixed observed cases permit exact text evidence; add the most widespread error.
    for rid in ['test_0168','test_0007','test_0027','test_0026','test_0028','test_0001']:
        chosen.append(next(x for x in tests if x['record_id']==rid))
    worst=max(tests,key=lambda x:sum(not x['methods'][m]['correct'] for m in PRIMARY))
    if worst['record_id'] not in {x['record_id'] for x in chosen}:chosen.append(worst)
    for i,row in enumerate(chosen):
        if i:r.heading(f"案例 {i+1}  {row['record_id']}",level=2,new_page=True)
        else:r.heading(f"案例 {i+1}  {row['record_id']}",level=2)
        r.p('專家答案：'+CHOICES[row['human_label']]+'。下列節錄均為來源原文；省略標記不屬於原文。',bold=True)
        a,b,note=legacy.differing_excerpts(row['A'],row['B'])
        r.p('參考開頭：'+legacy.excerpt(row['gold_claim'],220),size=9.1)
        r.p(note,size=9.1);r.p('A：'+a,size=9.1);r.p('B：'+b,size=9.1)
        ob=legacy_obs.get(row['record_id'])
        if ob:
            for e in ob.get('evidence',[]):assert e['quote'] in row[e['field']]
            r.p('原文觀察：'+ob['observation'],size=9.3)
        r.p('下表為自動指標的實際判斷。公開檔案未提供專家逐筆理由，不能把原文觀察當作專家判斷原因。',size=9.1)
        r.table(['方法','A分數','B分數','預測','是否一致'],[[LABELS[m],num(row['methods'][m]['score_a']),num(row['methods'][m]['score_b']),CHOICES[row['methods'][m]['prediction']],'是' if row['methods'][m]['correct'] else '否'] for m in PRIMARY],[4.6,3.4,3.4,3.0,3.0],font_size=8.1,number_columns=(1,2,3,4))
    r.heading('限制與研究解讀',new_page=True)
    for text in [
      '只有20個不同參考專利群組，原始比較列也存在重複。本文採分群bootstrap處理群內依賴，但小群組數與資料選取偏差仍限制推論。',
      'Quality為相對偏好標籤，不是逐候選的絕對品質分數；高文本相似度也不等同法律有效、專利可核准或內容完全正確。本報告只量化與既有專家答案的一致程度。',
      '所有測試參考與候選均曾出現在訓練。即使排除9筆完整重複比較，也不能消除此內容重疊；不可外推至全新專利。',
      '本次資料為英文專利請求項。繁體中文是報告語言，本次並未評估中文請求項或驗證跨語言效果。',
      'BERTScore、SimCSE、BARTScore、MoverScore與UniEval有不同輸入上限。標準設定下的結果反映模型與長文本處理的共同影響，並非只測量底層語義能力。',
      'PatClaimEval來自固定Quality checkpoint，其訓練來源與本資料相關；其他方法的通用預訓練／微調資料亦未全面排查專利重疊，不能聲稱完全沒有訓練污染。',
      '其餘四個人工構面未公開於本次資料，故不補造該四欄相關係數或正確率。G-Eval-4未執行，不把未執行當作零分參與排名。',
      '原論文沒有完整公開所有baseline設定。本次以固定且可檢查的設定執行，原論文數值用於外部參照，差距不能全歸因於某個單一實作因素。',
    ]:r.p(text)
    r.heading('驗證與重跑方式',new_page=True)
    r.p('交付前先從原始配對與CSV獨立重算偏好標籤、正確筆數、三分類混淆矩陣、Accuracy、Macro-F1及Weighted-F1，再核對JSON摘要。每筆分數與各方法輸出相符，全部1,228列與15,964個主要偏好判斷均完整，另有1,228個PatClaimEval長度變體判斷。')
    r.p('各方法checkpoint綁定輸入、程式、模型與環境設定；中斷續跑只使用已完整寫入且hash相符的紀錄，損毀的完整行、重複配對及未知ID均拒絕。Smoke parity與實際零新增推論續跑紀錄隨結果交付。')
    r.p('主入口為public-programs/verification/run_patent_ce_extended_evaluation.py。先執行prepare與preflight，再執行score、aggregate、verify及report；all依序執行。--methods可只評指定方法，--smoke使用獨立小樣本目錄，不混入正式checkpoint。')
    r.p('完整重跑：python public-programs/verification/run_patent_ce_extended_evaluation.py --stage all --output-dir outputs/新的執行目錄 --threads 4 --batch-size 1 --bootstrap-replicates 10000',size=9.0)
    r.p('只驗證既有輸出：python public-programs/verification/run_patent_ce_extended_evaluation.py --stage verify --output-dir outputs/'+out.name,size=9.0)
    r.p('模型、WSL環境及文件產生依賴須依README建立；權重、虛擬環境及逐頁檢查PNG不隨GitHub提供。Word／PDF使用bundled runtime與既有Word轉檔流程，重跑至其他主機需調整環境路徑。')
    r.heading('附件與來源')
    r.table(['附件','用途'],[['scores.csv／predictions.csv','完整原文、各方法分數、偏好、專家答案與正誤'],['summary.csv／results.json','各切分與群組統計、分類成效、相關、分群區間與成對差異'],['pair_metadata.csv／JSON','逐方法逐配對的token、截斷與片段資訊'],['*_scores.json／checkpoint','完整唯一配對分數、設定、來源與續跑資料'],['assets／environment／verification JSON','模型檔案hash、固定上游revision、環境與獨立驗證']],[6.3,11.1],font_size=9)
    for label,url in [
      ('Jiang, Scherz and Goetz 2025  Towards Better Evaluation for Generated Patent Claims','https://aclanthology.org/2025.acl-long.190/'),
      ('Patent-CE 原始資料與固定版本','https://github.com/scylj1/PatClaimEval/tree/8a9df74271be1330cf3b185c550e82513cf265f6'),
      ('Hugging Face evaluate','https://github.com/huggingface/evaluate'),
      ('BLEU 官方原始實作來源','https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py'),
      ('ROUGE 官方套件來源','https://github.com/google-research/google-research/tree/master/rouge'),
      ('NLTK METEOR 官方實作','https://www.nltk.org/api/nltk.translate.meteor_score.html'),
      ('BERTScore 官方程式','https://github.com/Tiiiger/bert_score'),
      ('BARTScore 官方程式','https://github.com/neulab/BARTScore'),
      ('MoverScore 官方程式','https://github.com/AIPHES/emnlp19-moverscore'),
      ('SimCSE 官方程式','https://github.com/princeton-nlp/SimCSE'),
      ('UniEval 官方程式','https://github.com/maszhongming/UniEval'),
      ('AlignScore 官方程式','https://github.com/yuh-zha/AlignScore'),
      ('CC BY-NC 4.0','https://creativecommons.org/licenses/by-nc/4.0/'),
    ]:r.link(label,url)
    docx=r.save()
    atomic_json(out/'report_build_manifest.json',dict(created_at=stamp(),source_results_sha256=digest(out/'results.json'),source_verification_sha256=digest(out/'verification.json'),builder_sha256=digest(__file__),docx_sha256=digest(docx),figures={p.name:digest(p) for p in figures.values()}))
    print(docx,flush=True)
    mark_artifact(out,'pdf')
    subprocess.run([sys.executable,str(Path(__file__).parents[1]/'patent_ce/render_report.py'),str(docx),'--output_dir',str(out/'qa'),'--emit_pdf'],check=True)
    shutil.copy2(out/'qa'/(NAME+'.pdf'),out/(NAME+'.pdf'))
    return docx

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir.resolve())
