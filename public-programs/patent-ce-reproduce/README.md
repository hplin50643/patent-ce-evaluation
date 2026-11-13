# 重現 Patent-CE 的 Longformer 與 ModernBERT 比較

使用論文的英文 Quality 原始切分，從 1,044 筆訓練記錄取出 105 筆驗證資料。兩種模型各用 seed 42 和 43 訓練 10 輪，在同一批 184 筆原始測試資料上評估。主要比較固定使用 seed 42 的第 10 輪權重，seed 43 與最低驗證 loss 權重另列於報告。

| Seed 42 第 10 輪 | Accuracy | Weighted F1 | Rank-pair Kendall τ-b | 離散 Spearman ρ |
| --- | ---: | ---: | ---: | ---: |
| Longformer | 59.24% | 55.39% | 0.4746 | 0.5544 |
| ModernBERT-base | 73.91% | 63.36% | 0.7653 | 0.8589 |

Accuracy 差值為 +14.67 個百分點，按 20 組參考請求項做成對 bootstrap 的 95% 區間為 +5.77 至 +25.74 個百分點。Weighted F1 的差值區間包含 0。ModernBERT 在主要種子僅判對 49 筆平手中的 1 筆。原始切分的訓練與測試內容有重疊。完整比較、平手診斷與兩種子的區間見[繁中報告](../../outputs/paper-modernbert/paper_modernbert_original_20261109/report.html)。

## 在 CPU 核對已發布的數字

從儲存庫根目錄執行。使用 Python 3.11 或 3.12 的虛擬環境。

```bash
python -m pip install -r public-programs/patent-ce-reproduce/requirements-cpu.txt
python public-programs/patent-ce-reproduce/verify-published.py
```

程式以原始資料重建切分，核對八組評估、1,472 列逐筆預測及兩個種子各 10,000 次 bootstrap。成功時輸出 `status: passed`。此流程使用 Git 中的原始資料、預測及設定即可運作。可用 `--json-out PATH` 保存核對結果。

報告資料包含權重 SHA-256，模型權重與訓練續跑狀態由 GPU 流程產生。前輪完整交付包可供已有該檔案的讀者載入歷史權重。歷史 `sha_manifest.json` 記錄完整訓練產物，CPU 核對的範圍是已發布的數值與資料來源。

## 從預訓練權重重新訓練

使用 Linux x86-64、Python 3.11 以上、Docker、NVIDIA Container Toolkit 與一張或兩張可用的 V100。Docker 映像會從固定 Python 3.11.12 基底安裝 PyTorch 2.7.1／CUDA 12.6。Longformer 用 Transformers 4.46.2，ModernBERT 用 4.50.3。映像建置與取得固定 Hugging Face 模型版本需連網，並請為映像、模型快取與兩輪權重預留磁碟空間。

```bash
bash public-programs/patent-ce-reproduce/run_training.sh ./reproduction-run --gpus 0,1
```

每次執行指定新的輸出目錄。只有一張 GPU 時傳 `--gpus 0`，兩個種子會依序訓練。程式先以論文公開設定與前輪記錄的補充設定訓練 Longformer，再將這次生成的 Longformer 權重和預測鎖定為 ModernBERT 的基準，最後產生 `reproduction-run/modernbert/report.html` 與獨立驗證 JSON。加入 `--author-replay` 可額外重評作者公開的 Longformer 權重。隨機訓練結果及權重雜湊可隨環境與執行而變化。

兩輪均使用 FP32、有效 batch 4、4096 tokens 上限、AdamW lr 5e-6、weight decay .01、成對排序 loss margin .1、tie tolerance .01。前輪 Longformer 保留預訓練 pooler，ModernBERT 使用原生 tokenizer 與新初始化的等價評分層。正式評估以嚴格 `abs(score_A-score_B) < 1e-4` 判定平手。兩個配置檔記錄作者未披露而由本實驗補定的設定。

| 路徑 | 內容 |
| --- | --- |
| `public-programs/paper-reproduction/` | Longformer 訓練、論文資料切分與評估 |
| `public-programs/paper-modernbert/` | ModernBERT 訓練、評分模型、成對統計與報告 |
| `public-programs/patent-ce-reproduce/` | Docker 訓練入口、基準鎖定與歷史數字核對 |

資料與作者 Longformer 推論程式保留上游 CC BY-NC 4.0 授權和姓名標示。ModernBERT 基礎模型依其 Apache 2.0 模型卡。來源見[資料與程式聲明](../../NOTICE.txt)、[作者推論程式來源](../paper-reproduction/author_source.json)、[PatClaimEval 論文](https://aclanthology.org/2025.acl-long.190/)和[ModernBERT 模型卡](https://huggingface.co/answerdotai/ModernBERT-base)。
