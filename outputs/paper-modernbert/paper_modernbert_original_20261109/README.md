# ModernBERT 與 Longformer 的原始切分比較

2026 年 11 月 9 日完成兩個模型、兩個種子各 10 輪的比較。每個模型與種子保留第 10 輪及驗證損失最低的結果，共八個條件，每個條件使用原始 184 筆測試資料。

- [繁體中文比較報告](report.html)
- [比較圖](comparison.png)及 [SVG](comparison.svg)
- [比較表](comparison.csv)及[完整統計](comparison.json)
- [1,472 筆測試預測](all_test_predictions.csv)
- [原始執行驗證](verify.json)及[原始交付雜湊清單](sha_manifest.json)

CPU 驗證會從儲存庫內的原始資料重建資料檔，核對原始切分與訓練設定，並用保存的逐筆分數重算兩個種子各 10,000 次成對群組 bootstrap。完整權重、tokenizer 與訓練產物可由重訓流程建立。原始交付雜湊清單也記錄了本次 Git 發布省略的檔案及其 SHA-256，其中 `baseline/` 指[前一輪 Longformer 的輸出目錄](../../paper-reproduction/paper_longformer_original_20261108)。

從儲存庫根目錄執行 CPU 驗證，環境安裝方式見[重現指南](../../../public-programs/patent-ce-reproduce/README.md)。

```sh
python public-programs/patent-ce-reproduce/verify-published.py
```

資料來源與使用條款見 [NOTICE](../../../NOTICE.txt)。
