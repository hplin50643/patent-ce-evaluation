# Patent CE 十三項方法評估與重跑說明

這個版本擴充原五項評估，使用公開 Patent-CE 的 **1,228 筆原始比較**：訓練集 1,044 筆、測試集 184 筆分開計算。13 種主要方法各產生 1,228 個偏好判斷，共 15,964 個；PatClaimEval 4096 另有 1,228 個敏感度判斷，總計 17,192 列長格式預測。

原五項評估及其檔案保留不變。新增版本放在 `outputs/patent_ce_extended_20261019_230412`；舊六種設定的保存分數經原始 manifest 與 SHA-256 核對後匯入，不重新推論或四捨五入。新八種方法只需評分 224 個不同的參考／候選配對，再回填為全部原始比較，保留重複列的權重。

## 方法與固定設定

| 方法 | 本次設定 |
| --- | --- |
| BLEU-1、BLEU-4 | Tokenizer13a，分別 `max_order=1`、`4`，不平滑，逐候選計算。 |
| ROUGE-1 F1、ROUGE-2 F1、ROUGE-L F1 | 逐候選 F1，不使用詞幹化。 |
| METEOR | NLTK tokenizer、WordNet；alpha 0.9、beta 3、gamma 0.5。 |
| BERTScore F1 | 原結果的 roberta-large 第 17 層；無 IDF、不作 baseline rescaling、slow tokenizer、512 tokens。 |
| BARTScore CNN | facebook/bart-large-cnn；gold 生成 candidate 的方向；來源與目標各上限 1,024 tokens；回報平均 log probability，通常為負數，越高越好。 |
| MoverScore v2 | distilbert-base-uncased、512 tokens、unigram；依固定訓練資料中去重的參考與候選群體分別建立 IDF，保留實際 IDF JSON。 |
| SimCSE | sup-simcse-bert-base-uncased、supervised CLS pooler、128 tokens、cosine。 |
| UniEval relevance | unieval-sum，只計 relevance；candidate 為 output、gold 為 reference、source 空字串，合併輸入上限 1,024。 |
| AlignScore large | large checkpoint、roberta-large、`nli_sp`，沿用官方分句及分塊聚合。 |
| PatClaimEval | 固定 Quality checkpoint、2,048 tokens 為主要設定；4,096 為補充設定。 |

G-Eval-4 **未執行**，只保留原論文引用數值，不用本機語言模型代替後沿用該名稱。所有本次推論都在本機 CPU 執行，不呼叫付費推論 API；環境建立及固定公開資產下載仍需要網路。

完整模型 revision、作者程式 commit、下載檔案 SHA-256、環境版本與 CPU／本機路徑相容調整，保存於 `assets.json`、`extended_environment_lock.json`、各 `*_scores.json` 的 `settings`、`provenance`、`environment`。權重及虛擬環境不放入報告壓縮檔或公開儲存庫。

## 真實答案與統計規則

原始 JSON 每筆只有 `gold_claim`、`A`、`B`、`human_eval.Quality`。`Quality=1` 表示 A 較佳，`0` 表示平手，`-1` 表示 B 較佳。公開檔案缺少其餘四個面向的逐筆人工標籤，因此本次實測只有 **Quality**；不補造 Completeness、Clarity、Consistency、Linkage 的正確率或相關係數。

每個方法分別算 `sA = score(gold, A)`、`sB = score(gold, B)`。令 `delta = sA - sB`：

```text
abs(delta) < 0.0001  → 平手 0
其餘 delta > 0      → A 較佳 1
其餘 delta < 0      → B 較佳 -1
```

等於正負門檻時仍保留勝負；全部運算使用未四捨五入值。這個規則沒有用訓練或測試答案調參。負的 BARTScore 同樣依高低比較，不取絕對值、不反轉方向。

Accuracy、錯誤率、三類 precision／recall／F1、Macro-F1、Weighted-F1 及混淆矩陣由偏好判斷對照專家答案計算。Macro-F1 固定平均 `[-1, 0, 1]` 三類；某類分母為 0 時該項記 0。非平手正確率與反向誤判率以專家非平手資料作分母。

相關係數分開保存：連續版本使用 `delta` 對專家標籤；離散版本使用預測標籤對專家標籤。Kendall 使用 tau-b，Spearman 使用套件的並列名次處理；任一輸入為常數時記 `null`，不假裝為零相關。原論文沒有充分說明其完整相關計算協定，因此本報告不宣稱精確重現原表數字。

95% 區間以 gold 參考文本群組作 bootstrap：seed 42、10,000 次；每個子集有 g 個群組，就有放回抽 g 次，保留抽中群組內全部比較。同一次抽樣共用於所有方法，產生 78 組主要方法成對差異及一組 PatClaimEval 長度差異。原始列有相依性，不採假設每列獨立的 Wilson 區間。

## 長文本與子集

每個方法使用自己的 tokenizer 與實際輸入結構判定截斷，不能拿 BERTScore token 數代替其他方法。逐配對 `metadata.truncated` 是明確布林值；沒有欄位時，統計層保留未知狀態，不自動視為未截斷。AlignScore 的官方分塊不等於丟棄全文，只有實際模型片段仍被截斷才記 True。

主分析保留全部訓練與測試資料；另外計算測試排除 9 筆訓練重複比較、測試排除 28 筆參考原文複製、訓練去除 20 筆完全重複，以及各方法的截斷／未截斷子集。空子集明列 `n=0`，統計值為不適用；未知截斷列不吸收進未截斷群組。截斷與未截斷群組的文本及難度不同，兩群差距不能當作截斷的因果效果。

## 環境建立與完整重跑

本次環境為 Windows 協調程式、WSL `Ubuntu-26.04` 新隔離推論環境，以及既有文件 runtime。新推論環境使用 CPython 3.10.20、CPU PyTorch 1.13.1、Transformers 4.26.1；完整含 hash 的套件鎖定檔為 `requirements-lock.txt`。`bootstrap.sh` 安裝固定 uv、建立獨立的 `~/.venvs/patent-ce-13methods-py310`，不修改原五方法的虛擬環境。

以下命令在專案根目錄的 PowerShell 執行。將新輸出名稱換成自己的執行目錄，勿指定原五方法資料夾。

```powershell
# 完整流程：prepare → preflight → 全部 Smoke → 全部正式 score → 補充樣本驗證 → aggregate → verify → report
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage all --output-dir .\outputs\patent_ce_extended_reproduction --threads 4 --batch-size 1 --bootstrap-replicates 10000
```

`--stage all` 會先對選取的全部新增方法完成 Smoke，再進入全部正式評分，不會把兩種結果混用。請勿另加 `--smoke`；`all --smoke` 會被拒絕。交付壓縮包要求全部八種新方法的 Smoke parity 與續跑證明。需要分段檢查時，可依序執行：

```powershell
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage prepare --output-dir .\outputs\patent_ce_extended_reproduction
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage preflight --output-dir .\outputs\patent_ce_extended_reproduction

# 全部八種方法的正式交付前檢查。
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage score --output-dir .\outputs\patent_ce_extended_reproduction --smoke --threads 4 --batch-size 1

# 全部八種新增方法的正式評分。
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage score --output-dir .\outputs\patent_ce_extended_reproduction --threads 4 --batch-size 1

# 選跑範例：小樣本寫入獨立 smoke/METHOD 子目錄，不混入正式結果。
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage score --output-dir .\outputs\patent_ce_extended_reproduction --methods bartscore simcse --smoke --threads 4 --batch-size 1

# 只執行指定的新方法；省略 --methods 時執行全部八種新增方法。
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage score --output-dir .\outputs\patent_ce_extended_reproduction --methods bartscore simcse --threads 4 --batch-size 1
```

八種新增方法識別碼為 `bleu1 rouge2_f1 meteor bartscore moverscore_v2 simcse unieval_relevance alignscore`。分次選取方法後，須等全部 14 種設定的完整分數檔都存在，才可進入聚合。若要指定已完成的其他資產清單，可於 score 階段加 `--assets <assets.json 路徑>`。

### 初始 Smoke 與短／中樣本補驗時序

初始 Smoke 實際選取「第一個參考原文複製配對、第一個非複製配對、原始文字總長度最大配對」，三者的 gold 加 candidate 字元數為 4,568、4,708、10,882；前兩者約位於第24與26百分位，不能稱為真正最短或中位樣本。各方法初始 Smoke 全數完成後，才進行全部八種方法的224配對正式評分。

全部八項正式評分及其零新增推論續跑證明完成後，再以獨立的 `supplemental_validation.py` 循序補驗真正最短與最接近中位數的兩個配對。字元數使用原始 Unicode 字元的 `len(gold_claim)+len(candidate)`，不正規化；距中位數相同時，選原 pairs 陣列較前者。本資料的最短配對共1,354字元（677＋677，亦為原文複製），中位數為5,752，選到最接近的5,746字元配對。最長與原文複製的初始證明一併保留。

補驗使用相同 frozen worker、模型資產、環境與評分設定；本次仍為 CPU threads=4、batch-size=1。補驗子輸入保留全部1,228筆 records，因此 MoverScore 仍使用相同20個參考、224個候選的訓練IDF，並核對IDF內容及群體hash。每法額外核對官方函式分數、同配對的正式分數、逐配對metadata與零新增推論resume。Align另外保存三個head對原始未省算forward的實測parity。

`--stage all` 在全部正式 score 後自動執行這項補驗。分段操作或單獨重跑補驗時，於WSL終端機執行以下命令；請先等正式模型全部完成，再使用相同threads及batch-size。

```bash
cd "/mnt/c/Users/agent/Project/專利挑戰"
/home/agent/.venvs/patent-ce-13methods-py310/bin/python public-programs/verification/patent_ce_extended/supplemental_validation.py \
  --input outputs/patent_ce_extended_reproduction/pairs.json \
  --output-dir outputs/patent_ce_extended_reproduction \
  --assets outputs/patent_ce_extended_reproduction/assets.json \
  --threads 4 --batch-size 1
```

補驗的subset輸入、Smoke、checkpoint及resume證明放在 `supplemental_sample_validation/` 獨立子目錄；run root的 `supplemental_sample_validation.json` 記錄原始／子輸入hash、選樣規則、實際長度、三種相同簽章、正式與官方parity、固定IDF及resume證明。只有八項全部通過，彙總狀態才是 `passed`；正式分數及checkpoint保持原檔。相同參數重跑時，已完成補驗使用worker原有的驗證續跑路徑。

每個補驗子程序以GNU `/usr/bin/time -v` 實際量測，原始resource log與hash一併保存。`methods.METHOD.resources` 記錄完整補驗子程序的wall time、user/system CPU time與maximum RSS（GNU time原欄位kbytes），範圍包含模型載入、兩個樣本、官方parity及resume證明；這不是正式224對推論的峰值。後續重跑另存resource log，保留初次量測，避免以零新增推論續跑時的低記憶體用量覆蓋模型載入時的實測峰值。GNU time的RSS也不當作同時所有子程序RAM的加總。

前置檢查會先確認WSL的 `/usr/bin/time` 可執行；乾淨Ubuntu若缺少，可先安裝發行版的 `time` 套件（例如 `sudo apt-get install time`），再執行preflight。

```powershell
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage aggregate --output-dir .\outputs\patent_ce_extended_reproduction --bootstrap-replicates 10000
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage verify --output-dir .\outputs\patent_ce_extended_reproduction
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\run_patent_ce_extended_evaluation.py --stage report --output-dir .\outputs\patent_ce_extended_reproduction
```

Word／PDF 產生使用原 Codex bundled Python、Microsoft Word COM、標準文件 renderer、Poppler 與 Microsoft JhengHei 字型。移至其他主機須依實際安裝位置調整 runtime 路徑；統計重算與結果驗證不需要重新下載模型。只執行 `verify.py` 使用 Python 標準函式庫及本資料夾的 `core.py`，不需 PyTorch。

## 中斷續跑與檢查

中斷後，使用相同輸出目錄、方法、執行參數、程式及資產清單重新執行 score；完整寫入的配對會從 checkpoint 回復。簽章綁定輸入、設定、程式、模型資產清單與環境，不相符時拒絕混用結果。只有最後一筆未寫完的紀錄可截掉；損毀完整行、重複配對與未知 ID 均視為錯誤。

完成後的 `*_resume_verification.json` 會實際走禁止新增推論的續跑路徑，確認保存分數及原檔 hash 不變。請勿在執行途中修改 worker 或覆寫 `assets.json`；要改設定就換新輸出目錄。Smoke 結果另保存與作者評分函式的 parity 檢查。

```powershell
.\LLM4DPCG\.venv\Scripts\python.exe -m unittest discover -s .\public-programs\verification\patent_ce_extended -p "test_*.py" -v
```

測試涵蓋手算混淆矩陣與 F1、嚴格平手邊界、負 BARTScore、候選交換、不等長群組重抽樣、常數相關、原始列回填與未知截斷。`verify.py` 另外獨立從 CSV 重算每筆偏好及所有子集的分類成效，並核對原始文本、專家標籤、配對 hash、各方法原始分數及完整 token／截斷 JSON。

## 產物與欄位

| 檔案 | 內容 |
| --- | --- |
| `scores.csv` | 1,228 筆完整 gold/A/B 原文、專家答案、原始列識別及各設定的分數與預測。 |
| `predictions.csv` | 17,192 列；`is_primary=True` 為 13 種主要方法，False 為 4096 敏感度設定。 |
| `summary.csv`、`results.json` | 切分／子集、分類成效、連續及離散相關、區間、成對差異與案例。 |
| `pair_metadata.csv`、`pair_metadata.json` | 3,136 筆方法／唯一配對資訊，保存分數與完整 worker metadata。 |
| `METHOD_scores.json` | 224 個唯一配對的實際分數、設定、來源、環境與完成標記。 |
| `METHOD_state.json`、`METHOD_checkpoint.jsonl` | 新方法可續跑狀態與逐配對紀錄。 |
| `moverscore_v2_idf.json` | 實際訓練語料 IDF 群體簽章與字詞權重。 |
| `verification.json` | 標準函式庫獨立核對結果與數值產物雜湊。 |
| `qa_review.json`、`report_build_manifest.json` | 最終文件逐頁檢查及文件／來源雜湊。 |
| `assets.json`、環境 lock、`THIRD_PARTY_NOTICES` | 固定來源與模型資產、套件、CPU 相容調整的重現紀錄。 |

CSV 使用 UTF-8 BOM 及標準多行引號，請使用 CSV reader 計算 record 數，勿用文字換行數計算。各方法欄位尾碼：`score_a`、`score_b`、`delta`、`prediction`、`correct`、`truncated`、`a_truncated`、`b_truncated`。分數不是品質正確機率，也不能用 `1-score` 當成專利內容錯誤比例。

`results.json` 的 `metadata.program_hashes` 是統計彙整當時的程式目錄快照。各次實際評分另綁定 `worker_sha256`；最終報告文字及排版版本以 `report_build_manifest.json` 的 `builder_sha256` 與交付清單為準，因為報告可在統計完成後繼續編排。

## 交付包與公開範圍

`report` 會產生 Word、PDF 與 `qa/page-N.png`。請實際檢視全部最終頁面，為每頁保存包含 `page`、`png_sha256`、`observation` 的檢查紀錄；每份紀錄另含 `reviewer`、`status="passed"`、最終 `docx_sha256` 與 `pdf_sha256`。使用文件 bundled Python 執行 `finalize_visual_review.py --output-dir 執行目錄 --reviews 檢查紀錄.json ...`，工具只核對完整覆蓋與版本，不代替人工視覺判斷。檔案或頁面改版後須重新檢視相應最終頁面。

只有獨立數值驗證通過、Word/PDF 最終 hash 與逐頁 QA 相符時，才可建立交付包：

```powershell
.\LLM4DPCG\.venv\Scripts\python.exe .\public-programs\verification\patent_ce_extended\package_results.py --output-dir .\outputs\patent_ce_extended_reproduction
```

壓縮檔為 `Patent_CE_十三項方法完整評估與重跑資料.zip`。`delivery_manifest.json` 以專案相對路徑列出每個檔案的 bytes 及 SHA-256；壓縮包也保留原 v1 manifest 的全部重現依賴。包內包含報告、評分與統計、配對 metadata、checkpoint／續跑證明、環境／資產清單及新舊程式，排除權重、虛擬環境、`__pycache__`、各版 QA 頁面 PNG、資產下載／錯誤日誌及先前 ZIP。

## 資料來源與解讀限制

資料來源為 [scylj1/PatClaimEval](https://github.com/scylj1/PatClaimEval)，固定 commit `8a9df74271be1330cf3b185c550e82513cf265f6`；上游宣告 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。原始資料及結果中重現的原文遵守姓名標示與非商用條件，不將所有第三方內容重新宣告為單一軟體授權。各新增方法的來源與固定版本見 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。

Jiang, Lekang; Scherz, Pascal A.; Goetz, Stephan. 2025. [Towards Better Evaluation for Generated Patent Claims](https://aclanthology.org/2025.acl-long.190/). ACL, 3775–3788. DOI 10.18653/v1/2025.acl-long.190。

1,228 筆是比較紀錄，只有 20 組不同參考文本。測試的 20 組參考及 186 個參考／候選配對都出現在訓練，另有 9 筆完整比較重複。因此結果描述官方切分下已見候選的專家偏好一致率，不能作為未見專利的泛化證據。公開資料也沒有逐筆專家理由；案例中的文字觀察不代表原始專家推理。

資料原文為英文專利請求項；繁體中文是報告語言，本次沒有評估中文請求項或跨語言效果。
