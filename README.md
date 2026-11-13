# Patent-CE 十三項方法評估

本儲存庫提供 Patent-CE 全部 **1,228 筆比較**的評估結果，訓練集 **1,044 筆**與測試集 **184 筆**分開呈現。主要評估 **13 種本機方法**；G-Eval-4 未執行，不呼叫付費 API。只對公開的 `human_eval.Quality` 專家答案計算成效。

## ModernBERT 與 Longformer 訓練比較

原始 184 筆 Quality 測試中，主要種子的 Accuracy 從 59.24% 提高至 73.91%。本次發布包含訓練程式、固定資料切分、逐筆預測與獨立數值核對。可從公開預訓練權重重新訓練兩種模型，也可在 CPU 上直接重算已發布結果。[重現指南](public-programs/patent-ce-reproduce/README.md) · [比較報告與結果](outputs/paper-modernbert/paper_modernbert_original_20261109/README.md)

## 最新報告與結果

- [完整繁體中文 PDF](outputs/patent_ce_extended_20261019_230412/Patent_CE_十三項方法評估與專家答案比較報告.pdf)
- [可編輯 Word 報告](outputs/patent_ce_extended_20261019_230412/Patent_CE_十三項方法評估與專家答案比較報告.docx)
- [完整報告與重跑資料 ZIP](outputs/patent_ce_extended_20261019_230412/Patent_CE_十三項方法完整評估與重跑資料.zip)
- [逐筆評分 CSV](outputs/patent_ce_extended_20261019_230412/scores.csv)／[偏好判斷 CSV](outputs/patent_ce_extended_20261019_230412/predictions.csv)／[摘要 CSV](outputs/patent_ce_extended_20261019_230412/summary.csv)
- [完整統計 JSON](outputs/patent_ce_extended_20261019_230412/results.json)／[欄位說明](outputs/patent_ce_extended_20261019_230412/data_dictionary.json)
- [獨立數值驗證](outputs/patent_ce_extended_20261019_230412/verification.json)／[逐頁排版驗收](outputs/patent_ce_extended_20261019_230412/qa_review.json)

## 測試集主要結果

正確率表示自動方法的「A 較佳、平手、B 較佳」判斷與專家一致的比例。分類 Macro-F1／Weighted-F1 與 ROUGE／BERTScore 的文本相似度 F1 為不同層次。

| 方法 | 正確筆數 | Accuracy | 分類 Macro-F1 | 分類 Weighted-F1 | 連續 τ-b | 連續 ρ | 離散 τ-b | 離散 ρ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BLEU-1 | 96 / 184 | 52.17% | 0.4022 | 0.4427 | 0.3521 | 0.4467 | 0.3472 | 0.3691 |
| BLEU-4 | 91 / 184 | 49.46% | 0.3811 | 0.4195 | 0.3475 | 0.4404 | 0.2874 | 0.3054 |
| ROUGE-1 F1 | 93 / 184 | 50.54% | 0.3915 | 0.4309 | 0.3534 | 0.4481 | 0.3176 | 0.3389 |
| ROUGE-2 F1 | 91 / 184 | 49.46% | 0.3817 | 0.4203 | 0.3542 | 0.4472 | 0.2881 | 0.3060 |
| ROUGE-L F1 | 94 / 184 | 51.09% | 0.3941 | 0.4338 | 0.3594 | 0.4545 | 0.3232 | 0.3436 |
| METEOR | 93 / 184 | 50.54% | 0.3891 | 0.4284 | 0.3448 | 0.4394 | 0.3115 | 0.3309 |
| BERTScore F1 | 88 / 184 | 47.83% | 0.3798 | 0.4143 | 0.3031 | 0.3884 | 0.2459 | 0.2624 |
| BARTScore CNN | 82 / 184 | 44.57% | 0.3401 | 0.3747 | 0.1879 | 0.2394 | 0.1744 | 0.1847 |
| MoverScore v2 | 89 / 184 | 48.37% | 0.3744 | 0.4122 | 0.3002 | 0.3831 | 0.2694 | 0.2875 |
| SimCSE | 85 / 184 | 46.20% | 0.3958 | 0.4220 | 0.2145 | 0.2746 | 0.2035 | 0.2193 |
| UniEval relevance | 91 / 184 | 49.46% | 0.3926 | 0.4282 | 0.2677 | 0.3433 | 0.2816 | 0.3005 |
| AlignScore large | 88 / 184 | 47.83% | 0.3702 | 0.4074 | 0.3237 | 0.4098 | 0.2569 | 0.2741 |
| PatClaimEval 2048 | 101 / 184 | 54.89% | 0.4495 | 0.4880 | 0.4895 | 0.6151 | 0.4289 | 0.4658 |

![測試集正確率與信賴區間](outputs/patent_ce_extended_20261019_230412/figures/test_accuracy_ci.png)

連續相關係數以 `score(A)-score(B)` 對專家答案計算；離散相關係數以三分類預測對專家答案計算。Kendall 使用 τ-b。兩種定義分開呈現。

## 解讀與資料限制

- 公開檔案只有 Quality 標籤；其他四個專利評估構面沒有逐筆真實答案，不填造相關或正確率。
- 測試集全部20組參考及186個不同參考／候選配對均曾出現在訓練集；其中9筆完整比較也重複。結果不能當作未見專利的泛化證據。
- 原始分數不是正確機率，也不能用 `1-score` 當內容錯誤百分比。BARTScore 為平均 token 對數機率，通常是負數，較高較佳。
- 本次資料為英文專利請求項；繁體中文是報告語言，未驗證中文請求項或跨語言效果。
- 所有方法保留原始分數尺度，以嚴格 `abs(delta)<0.0001` 判平手，沒有用測試答案調參。
- 信賴區間按參考專利分群重抽樣10,000次、seed42。方法間差異需參考成對區間，不能僅憑名次宣稱穩定優勢。
- 各方法的模型、tokenizer、截斷與分塊不同；完整揭露見報告及逐配對 metadata。MoverScore 明確使用 v2。

## 重現與檔案完整性

[完整執行說明](public-programs/verification/patent_ce_extended/README.md) 包含隔離 WSL Python 3.10.20 環境、固定模型與來源、各階段、方法選擇、續跑及文件產生依賴。新增八項方法均實際執行；原有五項及 PatClaimEval 長度變體，在資料、程式、設定與本機模型 hash 核對後，無損匯入已保存分數。

```powershell
python public-programs/verification/run_patent_ce_extended_evaluation.py --stage verify --output-dir outputs/patent_ce_extended_20261019_230412
```

完整推論需依 README 建立模型與環境；Word／PDF 產生使用原始 bundled runtime、Microsoft Word 及字型。權重、虛擬環境與中間頁面影像不隨儲存庫提供。`SHA256SUMS.txt` 與各次 `delivery_manifest.json` 可核對交付檔案。

## 前次五項評估

[前次 PDF 報告](outputs/patent_ce_evaluation_20261019_203847/Patent_CE_五項指標評估與專家答案比較報告.pdf) 與其原始資料、程式及驗證紀錄均保留。初次發布 commit 為 `c208682f95e4ddf835d69c36684deb2d39118851`。

## 來源與授權

Patent-CE 來自 [scylj1/PatClaimEval](https://github.com/scylj1/PatClaimEval)，固定 commit `8a9df74271be1330cf3b185c550e82513cf265f6`。上游宣告 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；原始資料及結果／報告中重現的原始文本仍遵循姓名標示與非商用條件。另見 [NOTICE](NOTICE.txt) 與[新增方法來源說明](public-programs/verification/patent_ce_extended/THIRD_PARTY_NOTICES)。

Jiang, Lekang; Scherz, Pascal A.; Goetz, Stephan. 2025. [Towards Better Evaluation for Generated Patent Claims](https://aclanthology.org/2025.acl-long.190/). ACL, 3775–3788. DOI: 10.18653/v1/2025.acl-long.190.
