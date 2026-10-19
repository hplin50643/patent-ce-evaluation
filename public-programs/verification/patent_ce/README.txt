Patent-CE 五項指標評估重跑說明

本程式評估公開 quality_train.json 與 quality_test.json，來源與模型版本均固定。
原始 JSON 不會修改。它不生成新候選、不重新訓練模型，也不調整平手門檻。

在 Windows PowerShell 切換至本專案根目錄後執行：

.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_evaluation.py --stage all --output-dir .\outputs\patent_ce_evaluation_新執行時間 --threads 4 --batch-size 4 --bootstrap-replicates 10000

--stage all 依序準備資料、計算四項相似度、以 WSL 計算 PatClaimEval 兩種長度、
進行統計、獨立驗證 CSV、製作 DOCX 及 PDF，並輸出逐頁 PNG 供人工檢查。

分段執行可將 all 換成 prepare、similarity、patclaim、aggregate、verify 或 report。
例如只重新彙整已完成的推論，可使用 --stage aggregate，再執行 verify 及 report。

若中斷，使用相同輸出目錄、程式與參數重跑。完成的唯一配對會從 JSONL 續跑紀錄讀取，
不會重新推論。輸入、模型、環境或設定不相符時會停止，不會混用不同來源的分數。
不要在推論過程中修改評分 worker、原始資料、模型或環境。

環境與固定模型

1. 四項相似度使用 LLM4DPCG/.venv（Windows CPU），模型位於其離線 Hugging Face 快取。
   roberta-large revision 為 722cf37b1afa9454edce342e7895e588b6ff1d59。
2. PatClaimEval 使用 WSL Ubuntu-26.04 的
   /home/agent/.venvs/patent-programs-py312/bin/python，模型讀取
   public-programs/models/PatClaimEval-Quality。
   固定 revision 為 c959e676f7eb61e32b0f8d9c4cc7d72a4aeb94d5。
3. 文件與圖表使用 Codex bundled Python；Word COM 輸出 PDF，canonical render_docx.py
   與 bundled Poppler 產出逐頁 PNG。環境需有 Microsoft Word 與正黑體。
4. results.json、各 worker JSON 與環境鎖定檔保存實際版本。模型權重及虛擬環境不隨報告壓縮檔提供。
   移至其他電腦前，須重建相同環境、快取和固定模型；原程式中的執行環境路徑可依新主機調整。

輸出內容

scores.csv：1,228 列。每列保留完整 gold/A/B 原文、專家標籤、六種設定的分數與偏好判斷。
predictions.csv：7,368 列，將各比較與六種設定攤開，適合統計分析。
summary.csv：各資料子集與方法的摘要。
results.json：主要統計、各類別結果、分群信賴區間、成對差異、敏感度與案例。
pairs.json：固定原始資料的準備結果與 224 個唯一候選/參考配對。
similarity_scores.json、patclaim_scores.json：唯一配對的完整推論、設定與模型來源資訊。
*_checkpoint.jsonl、*_progress.json：可續跑紀錄與進度。
verification.json：獨立由 CSV 重算並核對的結果。
qa/：DOCX 對應 PDF 與逐頁 PNG，供最後人工逐頁檢查。

指標與統計規則

BLEU-4：Tokenizer13a，max_order=4，smooth=False，逐候選計算。
ROUGE-1 F1、ROUGE-L F1：不使用詞幹化，逐候選計算 F1。
BERTScore F1：roberta-large 第17層、無IDF、不作baseline rescaling、slow tokenizer。
遵循標準512-token右側截斷，先計算未截斷長度並完整揭露受影響筆數。
PatClaimEval：原文輸入，官方 padding=max_length，分別計算2048及4096，全224配對均執行。

Δ = score(A) - score(B)。abs(Δ)<0.0001 判平手；否則依正負判A或B較佳。
專家答案 +1=A較佳，0=平手，-1=B較佳。不依測試標籤調參。
Macro-F1 固定三類平均；Weighted-F1 依各類支持筆數加權，分母為0時該項以0計。
95% 信賴區間按 gold_claim 群組重抽樣，10,000次，seed42，同一重抽樣共用於所有方法。
主train/test各20個gold群組；每個子集依實際g個群組有放回抽g次，保留群組內所有原始列。

驗證程式與測試

.\LLM4DPCG\.venv\Scripts\python.exe -m unittest discover -s .\public-programs\verification\patent_ce -p "test_*.py" -v

.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_evaluation.py --stage verify --output-dir .\outputs\本次目錄

獨立驗證會從 CSV 重新核對原文、標籤、分數、差值、嚴格平手門檻、逐欄截斷資訊、
所有子集的混淆矩陣、Accuracy、Macro-F1、Weighted-F1 及原始資料 SHA-256。
文件交付前另須逐頁視覺檢查，不能只因轉檔成功就認定排版正確。

解讀限制

公開資料只有整體品質 Quality 標籤，沒有各候選的絕對品質分數或逐筆專家判斷理由。
五項方法分數不是正確機率，也不能用1-score表示專利內容錯誤百分比。
訓練與測試的參考專利及候選文字全部重疊；測試中的9筆比較也與訓練相同。
因此本次評估反映官方切分中已見候選的偏好一致程度，不能當作未見專利的泛化證據。

來源

Jiang, Lekang; Scherz, Pascal A.; Goetz, Stefan (2025).
Towards Better Evaluation for Generated Patent Claims. ACL,3775–3788.
https://aclanthology.org/2025.acl-long.190/
https://github.com/scylj1/PatClaimEval/tree/8a9df74271be1330cf3b185c550e82513cf265f6
原始資料與上游程式庫授權為 CC BY-NC4.0。
