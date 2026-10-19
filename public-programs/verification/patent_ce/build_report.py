#!/usr/bin/env python3
"""Build the verified Traditional Chinese Patent-CE evaluation report.

Use the bundled Codex Python runtime.  This program reads completed evaluation
artifacts; it never changes source JSON or performs model inference.  The
documents artifact-operation marker must be run before calling this builder.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "Patent-CE"
REPORT_NAME = "Patent_CE_五項指標評估與專家答案比較報告"
METHOD_NAMES = {
    "bleu4": "BLEU-4",
    "rouge1_f1": "ROUGE-1 F1",
    "rougeL_f1": "ROUGE-L F1",
    "bertscore_f1": "BERTScore F1",
    "patclaimeval": "PatClaimEval",
}
LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "B 較佳", 0: "平手", 1: "A 較佳"}
COLORS = ["#33658A", "#4B796C", "#6B729B", "#8B674A", "#3E4D61"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value: float | None, digits: int = 2) -> str:
    return "不適用" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "不適用" if value is None else f"{value:.{digits}f}"


def interval(value: Any, percent: bool = True) -> str:
    if value is None:
        return "未計算"
    if isinstance(value, dict):
        value = [value.get("low", value.get("lower")), value.get("high", value.get("upper"))]
    formatter = pct if percent else num
    return f"{formatter(value[0])} 至 {formatter(value[1])}"


def excerpt(text: str, limit: int = 145) -> str:
    """Return a literal contiguous prefix; ellipsis is explicitly outside quote."""
    if len(text) <= limit:
        return "「" + text + "」"
    cut = text.rfind(" ", 0, limit)
    return "「" + text[:cut if cut > limit // 2 else limit] + "」〔後文略〕"


def local_excerpt(text: str, anchor: int, limit: int = 150) -> str:
    start = max(0, anchor - 45)
    end = min(len(text), start + limit)
    if start:
        boundary = text.find(" ", start, min(start + 24, anchor))
        if boundary >= 0:
            start = boundary + 1
    if end < len(text):
        boundary = text.rfind(" ", max(anchor + 15, end - 24), end)
        if boundary >= 0:
            end = boundary
    return ("〔前文略〕" if start else "") + "「" + text[start:end] + "」" + ("〔後文略〕" if end < len(text) else "")


def differing_excerpts(a: str, b: str) -> tuple[str, str, str]:
    changes = [item for item in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if item[0] != "equal"]
    content_changes = [item for item in changes if a[item[1]:item[2]].strip() or b[item[3]:item[4]].strip()]
    if not changes:
        return excerpt(a, 150), excerpt(b, 150), "A 與 B 原文完全相同"
    change = (content_changes or changes)[0]
    note = ("首個非空白差異" if content_changes else "首個空白格式差異") + f"附近  A 第 {change[1]+1} 字元  B 第 {change[3]+1} 字元"
    return local_excerpt(a, change[1]), local_excerpt(b, change[3]), note


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_plots() -> None:
    font_path = Path(r"C:\Windows\Fonts\msjh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "figure.dpi": 160,
                         "savefig.dpi": 190, "axes.spines.top": False, "axes.spines.right": False})


class Report:
    def __init__(self, output_dir: Path):
        self.doc = Document()
        self.output_dir = output_dir
        self.prose: list[str] = []
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(1.65)
        section.left_margin = section.right_margin = Cm(1.8)
        section.header_distance = Cm(0.6)
        section.footer_distance = Cm(0.65)
        for grid in section._sectPr.findall(qn("w:docGrid")):
            section._sectPr.remove(grid)
        for name, size, bold in [("Normal", 10.5, False), ("Title", 22, True),
                                  ("Subtitle", 12, False), ("Heading 1", 14, True),
                                  ("Heading 2", 11.5, True), ("Caption", 9, False)]:
            style = self.doc.styles[name]
            style.font.name, style.font.size, style.font.bold = "Calibri", Pt(size), bold
            style.font.color.rgb = RGBColor(0, 0, 0)
            rpr = style.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for key, value in {"ascii": "Calibri", "hAnsi": "Calibri", "eastAsia": "Microsoft JhengHei"}.items():
                fonts.set(qn("w:" + key), value)
            pf = style.paragraph_format
            pf.line_spacing = Pt(15.5 if name == "Normal" else size + 5)
            pf.space_after = Pt(6)
            pf.widow_control = True
            if name.startswith("Heading"):
                pf.space_before, pf.keep_with_next = Pt(9), True
        for style in self.doc.styles:
            for border in style.element.xpath(".//w:pBdr"):
                border.getparent().remove(border)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer.add_run("Patent CE 評估報告  |  第 ").font.size = Pt(8)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
        footer.add_run(" 頁").font.size = Pt(8)
        self.doc.core_properties.title = REPORT_NAME
        self.doc.core_properties.subject = "英文專利請求項評估指標與人工品質偏好的一致性"
        self.doc.core_properties.author = ""
        self.doc.core_properties.keywords = "Patent-CE, BLEU-4, ROUGE, BERTScore, PatClaimEval"

    def p(self, text: str, *, size: float | None = None, bold: bool = False, after: float = 6) -> Any:
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(after)
        run = paragraph.add_run(text)
        run.bold = bold
        if size:
            run.font.size = Pt(size)
            paragraph.paragraph_format.line_spacing = Pt(size + 4)
        self.prose.append(text)
        return paragraph

    def heading(self, text: str, *, level: int = 1, new_page: bool = False) -> None:
        paragraph = self.doc.add_paragraph(text, style=f"Heading {level}")
        if new_page:
            paragraph.paragraph_format.page_break_before = True
        self.prose.append("\n" + "#" * level + " " + text + "\n")

    def table(self, headers: list[str], rows: list[list[Any]], widths: list[float] | None = None,
              *, font_size: float = 9.1, number_columns: tuple[int, ...] = ()) -> None:
        widths = widths or [17.4 / len(headers)] * len(headers)
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        borders = OxmlElement("w:tblBorders")
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            border = OxmlElement("w:" + edge)
            for attr, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
                border.set(qn("w:" + attr), value)
            borders.append(border)
        table._tbl.tblPr.append(borders)
        for i, width in enumerate(widths):
            table.columns[i].width = Cm(width)
        for i, text in enumerate(headers):
            table.rows[0].cells[i].text = str(text)
        for values in rows:
            for cell, text in zip(table.add_row().cells, values):
                cell.text = str(text)
        for row_index, row in enumerate(table.rows):
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement("w:cantSplit"))
            if row_index == 0:
                props.append(OxmlElement("w:tblHeader"))
            for col_index, cell in enumerate(row.cells):
                cell.width = Cm(widths[col_index])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "344C60" if row_index == 0 else ("F1F4F6" if row_index % 2 == 0 else "FFFFFF"))
                tcpr.append(shading)
                margins = OxmlElement("w:tcMar")
                for edge in ["top", "bottom", "left", "right"]:
                    margin = OxmlElement("w:" + edge)
                    margin.set(qn("w:w"), "70" if edge in ["top", "bottom"] else "95")
                    margin.set(qn("w:type"), "dxa")
                    margins.append(margin)
                tcpr.append(margins)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = Pt(font_size + 3.2)
                    if col_index in number_columns:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.size = Pt(font_size)
                        if row_index == 0:
                            run.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
        self.prose.append("| " + " | ".join(headers) + " |")
        self.prose.append("| " + " | ".join(["---"] * len(headers)) + " |")
        self.prose.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def figure(self, path: Path, caption: str, width: float = 6.65) -> None:
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(path), width=Inches(width))
        self.doc.add_paragraph(caption, style="Caption")
        self.prose.append(f"![{caption}](figures/{path.name})")

    def link(self, label: str, url: str, text: str = "") -> None:
        paragraph = self.p(text, size=9.2, after=5)
        relation = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
        element = OxmlElement("w:hyperlink")
        element.set(qn("r:id"), relation)
        run = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "24527A")
        rpr.append(color)
        run.append(rpr)
        content = OxmlElement("w:t")
        content.text = label
        run.append(content)
        element.append(run)
        paragraph._p.append(element)
        self.prose.append(f"[{label}]({url})")

    def save(self) -> Path:
        path = self.output_dir / (REPORT_NAME + ".docx")
        self.doc.save(path)
        (self.output_dir / "report_content.md").write_text("\n\n".join(self.prose), encoding="utf-8")
        return path


def metric_label(method: str, results: dict[str, Any]) -> str:
    return results.get("method_labels", {}).get(method, METHOD_NAMES.get(method, method))


def primary_table(results: dict[str, Any], split: str = "test") -> list[list[Any]]:
    stats = results["results"][split]
    return [[metric_label(method, results), f"{stats[method]['correct']} / {stats[method]['n']}",
             pct(stats[method]["accuracy"]), interval(stats[method].get("ci", {}).get("accuracy")),
             num(stats[method]["macro_f1"]), pct(stats[method]["non_tie_accuracy"])]
            for method in results["primary_methods"]]


def make_figures(results: dict[str, Any], directory: Path) -> dict[str, Path]:
    setup_plots()
    directory.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    methods = results["primary_methods"]
    test = results["results"]["test"]
    labels = [metric_label(method, results) for method in methods]
    fig, ax = plt.subplots(figsize=(8.3, 3.3), constrained_layout=True)
    accuracies = np.array([test[method]["accuracy"] for method in methods])
    cis = [test[method].get("ci", {}).get("accuracy", [test[method]["accuracy"]] * 2) for method in methods]
    xerr = np.array([[accuracies[i] - cis[i][0], cis[i][1] - accuracies[i]] for i in range(len(methods))]).T
    positions = np.arange(len(methods))
    ax.barh(positions, accuracies * 100, color=COLORS[:len(methods)], height=.58)
    ax.errorbar(accuracies * 100, positions, xerr=np.maximum(0, xerr * 100), fmt="none", ecolor="#222222", capsize=3)
    majority = results["baselines"]["majority"]["test"]["accuracy"]
    ax.axvline(majority * 100, linestyle="--", linewidth=1.1, color="#9A5D4A", label=f"多數類基準 {majority*100:.2f}%")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("測試集三分類正確率  %")
    ax.set_xlim(0, min(100, max(85, max(ci[1] for ci in cis) * 100 + 14)))
    for index, accuracy in enumerate(accuracies):
        ax.text(accuracy*100 + 1.2, index - .24, f"{accuracy*100:.2f}%", fontsize=9)
    ax.legend(loc="lower right", fontsize=8.7)
    ax.grid(axis="x", alpha=.14)
    figures["accuracy"] = directory / "test_accuracy_ci.png"
    fig.savefig(figures["accuracy"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.4), constrained_layout=True)
    for index, method in enumerate(methods):
        ax = axes.flat[index]
        matrix = np.asarray(test[method]["confusion_matrix"], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=13,
                        color="white" if matrix[row, col] > matrix.max()*.60 else "#202020")
        ax.set_xticks(range(3), ["B 較佳", "平手", "A 較佳"], fontsize=9)
        ax.set_yticks(range(3), ["B 較佳", "平手", "A 較佳"], fontsize=9)
        ax.set_title(metric_label(method, results), fontsize=11)
        ax.set_xlabel("指標預測", fontsize=9)
        ax.set_ylabel("專家答案", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes.flat[5].axis("off")
    axes.flat[5].text(.04, .92, "讀法", fontsize=12, fontweight="bold", va="top", transform=axes.flat[5].transAxes)
    axes.flat[5].text(.04, .78, "對角線為預測正確\n非對角線為預測錯誤\n每格為原始比較筆數\n標籤順序固定為 -1  0  1", fontsize=10, linespacing=1.7, va="top", transform=axes.flat[5].transAxes)
    figures["confusion"] = directory / "test_confusion_five_metrics.png"
    fig.savefig(figures["confusion"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 3.3), constrained_layout=True)
    gold_ties = test[methods[0]]["classes"]["0"]["support"]
    predicted = [sum(np.asarray(test[m]["confusion_matrix"])[:, 1]) for m in methods]
    correct = [test[m]["confusion_matrix"][1][1] for m in methods]
    positions = np.arange(len(methods))
    ax.bar(positions-.18, predicted, width=.35, label="預測為平手", color="#718DA1")
    ax.bar(positions+.18, correct, width=.35, label="其中正確平手", color="#355F55")
    ax.axhline(gold_ties, color="#965D4A", linestyle="--", label=f"專家平手 {gold_ties} 筆")
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_ylabel("比較筆數")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.20))
    ax.grid(axis="y", alpha=.13)
    figures["ties"] = directory / "test_tie_predictions.png"
    fig.savefig(figures["ties"], bbox_inches="tight")
    plt.close(fig)

    raw = read_json(DATA / "quality_test.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    lengths = [[len(row[key].split()) for row in raw] for key in ["gold_claim", "A", "B"]]
    boxes = axes[0].boxplot(lengths, tick_labels=["參考文本", "候選 A", "候選 B"], patch_artist=True)
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(.65)
    axes[0].set_ylabel("依空白切分的詞數")
    axes[0].set_title("測試集完整文本長度", fontsize=11)
    truncation_count = results["audit"]["truncation"]["test"]["patclaim_rows"]
    axes[1].bar(["未截斷比較", "含截斷比較"], [len(raw)-truncation_count, truncation_count], color=["#718DA1", "#9C7054"])
    for i, count in enumerate([len(raw)-truncation_count, truncation_count]):
        axes[1].text(i, count+2, str(count), ha="center")
    axes[1].set_ylim(0, len(raw)*1.12)
    axes[1].set_ylabel("比較筆數")
    axes[1].set_title("PatClaimEval 2048 tokens", fontsize=11)
    figures["length"] = directory / "text_length_and_truncation.png"
    fig.savefig(figures["length"], bbox_inches="tight")
    plt.close(fig)
    return figures


def stat_class_table(results: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for method in results["primary_methods"]:
        stat = results["results"]["test"][method]
        for label in LABELS:
            item = stat["classes"][str(label)]
            rows.append([metric_label(method, results), LABEL_NAMES[label], item["support"],
                         num(item["precision"], 3), num(item["recall"], 3), num(item["f1"], 3)])
    return rows


def compact_settings(settings: dict[str, Any], prefix: str = "") -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in settings.items():
        name = prefix + key
        if isinstance(value, dict):
            rows.extend(compact_settings(value, name + "."))
        elif isinstance(value, (list, tuple)):
            if len(value) <= 12:
                rows.append([name, json.dumps(value, ensure_ascii=False)])
        elif value is not None and len(str(value)) <= 220:
            rows.append([name, str(value)])
    return rows


def validate_report_inputs(results: dict[str, Any], source_dir: Path) -> None:
    """Guard document-specific assumptions before any artifact is authored."""
    for required in ["metadata", "results", "primary_methods", "baselines", "cases", "verification"]:
        if required not in results:
            raise ValueError("Missing completed-results field: " + required)
    if results["verification"].get("status") != "passed":
        raise ValueError("The independent evaluation verification must pass before report authoring.")
    expected_methods = list(METHOD_NAMES)
    if results["primary_methods"] != expected_methods:
        raise ValueError("The report is scoped to the five agreed primary metrics in their fixed order.")
    if len(results["cases"]) != 8 or len({case["record_id"] for case in results["cases"]}) != 8:
        raise ValueError("Eight distinct auditable example records are required.")
    raw_test = read_json(DATA / "quality_test.json")
    for case in results["cases"]:
        if case["split"] != "test" or not 1 <= case["row_number"] <= len(raw_test):
            raise ValueError("Example record is outside the official test split.")
        original = raw_test[case["row_number"]-1]
        if any(case[key] != original[key] for key in ["gold_claim", "A", "B"]) or case["human_label"] != original["human_eval"]["Quality"]:
            raise ValueError("An example excerpt source differs from its original JSON row.")
        for method in expected_methods:
            if method not in case["methods"]:
                raise ValueError("Example is missing a primary method: " + method)
    if not (source_dir / "comparisons.json").is_file():
        raise FileNotFoundError("comparisons.json is required for the literal long-input audit table.")
    for split, count in [("train", 1044), ("test", 184)]:
        if any(results["results"][split][method]["n"] != count for method in expected_methods):
            raise ValueError("Main tables must retain every original row in the pinned split.")


def build_report(results: dict[str, Any], output_dir: Path, source_dir: Path | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(results, output_dir / "figures")
    report = Report(output_dir)
    test = results["results"]["test"]
    train = results["results"]["train"]
    methods = results["primary_methods"]
    winner = max(methods, key=lambda method: test[method]["accuracy"])
    winner_name = metric_label(winner, results)
    pat = test["patclaimeval"]
    majority = results["baselines"]["majority"]["test"]["accuracy"]
    comparison_path = (source_dir or output_dir) / "comparisons.json"
    comparisons = read_json(comparison_path)
    test_comparisons = [row for row in comparisons if row["split"] == "test"]
    date_text = str(results["metadata"].get("created_at_taipei", "2026-10-19"))
    methods_header = ["評估方法", "正確筆數", "三分類正確率", "95% 群組信賴區間", "Macro F1", "非平手正確率"]
    widths = [3.05, 1.75, 2.25, 4.0, 2.0, 2.35]

    # 1. An outcome-first executive opening, followed by the actual result table.
    report.doc.add_paragraph("Patent CE 五項指標評估報告", style="Title")
    report.doc.add_paragraph("與專家品質偏好及參考請求項的差距分析", style="Subtitle")
    report.p("完成日期  " + date_text[:10] + "    報告語言  繁體中文", size=9.5, after=13)
    report.p(f"在官方測試集 184 筆比較中，{winner_name} 的三分類正確率點估計最高，為 {pct(test[winner]['accuracy'])}"
             f"（{test[winner]['correct']} 筆正確）。PatClaimEval 的正確率為 {pct(pat['accuracy'])}，"
             f"Macro F1 為 {num(pat['macro_f1'])}。此處的正確率，是指標對 A 較佳、平手、B 較佳的判斷與專家標籤相同的比例。", bold=True)
    report.p("本報告以 BLEU-4、ROUGE-1 F1、ROUGE-L F1、BERTScore F1 與 PatClaimEval，"
             "分別比較每份候選請求項與 gold_claim，接著檢查兩份候選的分數高低能否重現專家答案。"
             "全數 1,228 筆資料均納入；1,044 筆訓練資料與 184 筆測試資料分開呈現。")
    report.table(methods_header, primary_table(results), widths, number_columns=(1, 2, 3, 4, 5))
    report.p("研究解讀的範圍", bold=True)
    report.p("官方測試集的 20 組參考文本及全部 186 組參考文本與候選配對，都曾出現在訓練集，"
             "其中 9 筆完整比較亦重複。結果反映官方切分中已見候選的比較能力，無法據此估計對新專利或新候選文本的泛化表現。")
    report.p(f"所有測試資料都預測為 B 較佳的多數類基準為 {pct(majority)}。本次另檢查各類別的誤判、"
             "排除重複比較及複製參考文本的敏感度、長文本截斷、成對差異的信賴區間，並提供 8 筆原文案例。")
    report.p("閱讀路徑  第 2 至 5 節說明資料與方法；第 6 至 13 節分析結果；第 14 至 17 節展示案例；"
             "最後一節提供重現與來源資訊。", size=9.5)

    # 2. Dataset and the two distinct forms of ground truth.
    report.heading("二  資料內容與真實答案", new_page=True)
    report.p("Patent-CE 研究專利請求項品質評估。每筆資料包含同一參考文本下的兩組候選請求項，"
             "以及專家對兩者整體品質的相對判斷。每組文本可能包含多項請求項，因此一列比較並非一項請求項，也未必代表一件不同專利。 [1][2]")
    report.table(["欄位", "本次用途"], [
        ["gold_claim", "原始參考請求項文本，供五項指標分別評分 A 與 B。"],
        ["A 與 B", "待評估的兩組完整候選文本，維持 JSON 原始順序。"],
        ["human_eval.Quality", "專家偏好答案：1 表示 A 較佳，0 表示平手，−1 表示 B 較佳。"],
    ], [4.5, 12.9])
    report.table(["切分", "比較筆數", "A 較佳", "平手", "B 較佳", "不同參考文本"], [
        ["訓練", 1044, 355, 333, 356, 20], ["測試", 184, 67, 49, 68, 20],
        ["合計", 1228, 422, 382, 424, 20],
    ], [2.2, 2.8, 2.8, 2.8, 2.8, 4.0], number_columns=(1, 2, 3, 4, 5))
    report.p("兩種答案回答不同問題。gold_claim 用來衡量候選與參考的文字或語意接近程度；"
             "human_eval.Quality 用來衡量評估器是否選到專家較偏好的版本。候選分數與專家標籤的差不能相減，"
             "因為前者為相似度或模型品質分數，後者是 −1、0、1 的相對偏好類別。")
    report.heading("資料重複與切分重疊", level=2)
    report.table(["檢查", "結果", "對解讀的影響"], [
        ["訓練集完全重複列", "20 列", "原樣保留；主要訓練結果依官方列權重。"],
        ["測試集完全重複列", "0 列", "測試集內沒有完全相同的重複比較。"],
        ["測試參考文本曾在訓練出現", "20 / 20 組", "無法證明對未見參考文本的泛化。"],
        ["測試候選配對曾在訓練出現", "186 / 186 組", "每個測試候選都屬已見候選。"],
        ["測試比較曾在訓練出現", "9 / 184 列", "另看排除此 9 列的 175 列結果。"],
    ], [6.3, 3.0, 8.1], font_size=9.0)
    report.p("對同一份 gold 及相同原始 A、B 文本，以交換順序並同步反轉標籤的方式檢查，未發現互相矛盾的偏好標註。"
             "這項檢查只確認公開檔案內部一致，不能證明每筆標註的判斷理由。", size=9.5)

    # 3. Exact reproducible definitions for the lexical metrics.
    report.heading("三  文字相似度指標與公式", new_page=True)
    report.p("對每筆比較，先計算 sA = metric(gold_claim, A) 與 sB = metric(gold_claim, B)，"
             "再計算 Δ = sA − sB。五項方法都依較高分表示較接近參考或較受模型偏好的方向解讀。"
             "各方法的原始尺度不同，僅比較同一方法的 A 與 B。")
    report.heading("BLEU 四階分數", level=2)
    report.p("BLEU-4 衡量候選與參考之間 1 至 4 階連續詞組的修正精確率，並以長度懲罰抑制過短候選。"
             "本次使用逐候選的 sentence BLEU-4，因此可對每一筆 A/B 比較產生分數；它與把所有文字合併計算的 corpus BLEU 不同。 [3]")
    report.p("BLEU-4 = BP × exp[(log p1 + log p2 + log p3 + log p4) / 4]", bold=True)
    report.p("其中 pn 為裁切計數後的 n-gram 精確率；若候選長度 c 大於參考長度 r，BP = 1；"
             "否則 BP = exp(1 − r/c)。本次使用 evaluate 的 BLEU、Tokenizer13a、max_order = 4、smooth = False；"
             "若某階精確率為 0，未平滑的 BLEU 結果為 0。")
    report.p("高 BLEU 表示較多相同的連續字詞，不能單獨證明技術特徵完整。若改寫保留技術意義但改變用詞或語序，BLEU 仍可能降低。")
    report.heading("ROUGE 一階與最長共同子序列", level=2)
    report.p("ROUGE-1 以 unigram 重疊計數計算 precision 與 recall，再取 F1；ROUGE-L 以最長共同子序列"
             "（LCS）長度計算對應的 precision、recall 與 F1。LCS 允許字詞在中間被其他文字隔開，但保留先後順序。 [4]")
    report.p("P = overlap / 候選詞數    R = overlap / 參考詞數", bold=True)
    report.p("ROUGE F1 = 2PR / (P + R)；當分母為 0 時依套件定義處理。", bold=True)
    report.p("ROUGE-L 將 overlap 改為 LCS 長度。本次回報 ROUGE-L F1，並非把各句重組的 ROUGE-Lsum。"
             "F1 同時考慮候選中有多少詞獲參考支持，以及參考中有多少詞被候選覆蓋。"
             "本次使用 use_stemmer = False、use_aggregator = False，直接保存逐候選 fmeasure。")
    report.heading("指標共同限制", level=2)
    report.p("字詞重疊可快速指出候選與參考的表面差異，卻難以判斷省略的字詞是否屬必要技術特徵、"
             "關係用語是否改變保護範圍，或兩段近似文字的概念連結是否正確。報告中的 accuracy 是額外對照專家答案後取得，並非 BLEU 或 ROUGE 自帶的百分比。")

    # 4. Semantic and task-specific models, both with token-length limitations.
    report.heading("四  語意模型與專利品質模型", new_page=True)
    report.heading("BERTScore F1", level=2)
    report.p("BERTScore 先以預訓練語言模型產生 token 的情境向量，再讓每個候選 token 找到參考中"
             "餘弦相似度最高的 token。以候選方向聚合得到 precision，以參考方向聚合得到 recall，最後取 F1。"
             "本次使用實際設定中的模型、層數與正規化選項，而不把不同 BERTScore 設定視為可互換的同一分數。 [5]")
    report.p("PBERT = 候選各 token 對參考 token 的最大餘弦相似度之平均", bold=True)
    report.p("RBERT = 參考各 token 對候選 token 的最大餘弦相似度之平均", bold=True)
    report.p("BERTScore F1 = 2 × PBERT × RBERT / (PBERT + RBERT)", bold=True)
    report.p("本次設定為 roberta-large 第 17 層、英文、idf = False、rescale_with_baseline = False、"
             "use_fast_tokenizer = False，保留模型原始 512 tokens 右側截斷。BERTScore 套件會自行去除首尾空白；"
             "此為套件內部行為，原始資料與輸出文本仍維持原樣。", size=9.5)
    report.p("語意相似的改寫有機會取得較高分，但 token 配對不保證完整保留請求項中的限定條件與元件關係。"
             "模型的最大 token 長度會限制實際可見內容，截斷的影響另於長文本章節分析。")
    report.heading("PatClaimEval 整體品質模型", level=2)
    report.p("本次使用 lj408/PatClaimEval-Quality。它將參考與候選編碼為一組 sentence pair，經 Longformer、"
             "線性層與 sigmoid 產生 0 至 1 的分數。分別評分 A 與 B 後，再比較分數差。"
             "公開資料及本次模型僅涵蓋整體品質 Quality；其餘四個專利品質面向沒有在本次另行評估。 [1][2]")
    report.p("s = sigmoid(Linear(Longformer(gold_claim, candidate)))", bold=True)
    report.p("主要分析依官方 score_pair 預設 max_length = 2048，對超長輸入執行 tokenizer 截斷；"
             "另以 4096 的結果作敏感度分析。輸出 0.8 表示模型在此輸入下給出的相對品質分數，不能解讀成 80% 正確率或 80% 專家同意機率。")
    report.heading("尺度與完整文本", level=2)
    report.p("BLEU、ROUGE、BERTScore 與 PatClaimEval 的 0.8 沒有相同意義；本次不把五項原始分數直接相加、平均或相減。"
             "比較方法優劣時，以相同 184 筆的專家一致率及其差異為依據。所有方法沿用同一份原始文本，"
             "各自的分詞器仍可能忽略或保留不同的空白與格式差異。")

    # 5. Decision rule and the distinction between raw scores and accuracy.
    report.heading("五  分數如何轉成答案與正確率", new_page=True)
    report.p("每個指標先得到 Δ = sA − sB，再以同一個預先固定的 ε = 0.0001 判定平手。"
             "這是避免將極小浮點差異直接視為勝負的保守規則，沒有利用訓練或測試標籤挑選門檻。"
             "五項方法尺度不同，共用門檻是一項固定評測約定，並非經驗證的通用品質相當標準。")
    report.p("預測 ŷ = 0  若 |Δ| < 0.0001；否則 Δ > 0 時 ŷ = 1，Δ < 0 時 ŷ = −1。", bold=True)
    report.p("等於 +0.0001 時選 A，等於 −0.0001 時選 B；恰好落在邊界不判平手。"
             "全部運算使用未四捨五入分數；表格顯示成相同小數值的兩個分數，仍可能產生非平手預測。", size=9.5)
    report.p("Accuracy = 預測標籤與專家標籤相同的筆數 / 全部比較筆數", bold=True)
    report.p("Error rate = 1 − Accuracy", bold=True)
    report.p("Precisionk = TPk / (TPk + FPk)    Recallk = TPk / (TPk + FNk)", bold=True)
    report.p("F1k = 2 × Precisionk × Recallk / (Precisionk + Recallk)", bold=True)
    report.p("Macro F1 = 三類 F1 的算術平均；Weighted F1 = 依各類實際筆數加權的 F1。", bold=True)
    report.p("此處的分類 F1 衡量 A 較佳、平手、B 較佳的預測品質。它與 ROUGE-1 F1、ROUGE-L F1、"
             "BERTScore F1 名稱中的文本相似度 F1 是不同層次的數值，不能互相替代。", size=9.3)
    report.p("Macro F1 讓三種類別各占相同權重，能揭露某方法只擅長選出 A 或 B、卻幾乎抓不到平手的情況。"
             "Weighted F1 會受到類別占比影響。本次兩者均保存，主要表格以 Macro F1 為主；數值以 0 至 1 表示。"
             "Precision、recall 或 F1 的分母為 0 時，該值依固定三類規則記為 0。")
    report.heading("補充統計與不確定性", level=2)
    report.p("非平手正確率只看專家答案為 A 或 B 的資料；若指標在這些資料判平手，仍算錯誤。"
             "反向誤判指專家選 A 而指標選 B，或專家選 B 而指標選 A，這類錯誤與把勝負判成平手分開記錄。")
    report.p("反向誤判率的分母為專家非平手的比較筆數，測試集為 135 筆；其餘三分類正確率與錯誤率的分母為全部 184 筆。", size=9.3)
    report.p(f"信賴區間以 gold_claim 的 20 個群組為重抽樣單位，執行 {results['metadata'].get('bootstrap_replicates', 10000):,} 次 bootstrap，"
             f"亂數種子為 {results['metadata'].get('seed', 42)}。每次抽到群組時保留其中全部比較，"
             "以維持同一參考文本下的候選相依性。95% 區間使用重抽樣分布的第 2.5 與 97.5 百分位。")
    report.p("每個分析子集若只有 g 組實際出現的 gold，就從該 g 組有放回抽取 g 次。"
             "本次未採把每列視為獨立試驗的 Wilson 區間，以免忽略共享參考與候選造成的相依。", size=9.3)
    report.p("群組數只有 20，區間反映這批公開參考文本的變異，不能補足獨立專利數量有限及訓練測試重疊的限制。"
             "方法間差異另採同一次群組重抽樣計算成對信賴區間，以保留兩者在相同資料上的關聯。")

    # 6. Main held-out-pair result.
    report.heading("六  官方測試集的整體表現", new_page=True)
    report.table(methods_header, primary_table(results), widths, number_columns=(1, 2, 3, 4, 5))
    report.figure(figures["accuracy"], "圖 1  官方測試集正確率與 95% 群組 bootstrap 信賴區間")
    report.table(["方法", "Macro F1", "Macro F1 的 95% 信賴區間", "Weighted F1"], [
        [metric_label(m, results), num(test[m]["macro_f1"]),
         interval(test[m]["ci"]["macro_f1"], percent=False), num(test[m]["weighted_f1"])]
        for m in methods], [4.2, 3.0, 6.5, 3.7], font_size=8.8, number_columns=(1, 2, 3))
    report.p(f"{winner_name} 在這批測試比較中取得最高點估計 {pct(test[winner]['accuracy'])}，"
             f"相較多數類基準高 {(test[winner]['accuracy']-majority)*100:.2f} 個百分點。"
             "最高點估計只描述本次資料；是否有足夠證據認定方法間存在差異，需查看成對信賴區間。")
    report.p(f"PatClaimEval 判對 {pat['correct']} 筆、判錯 {pat['n']-pat['correct']} 筆，錯誤率為 {pct(pat['error_rate'])}。"
             f"其正確率 95% 信賴區間為 {interval(pat.get('ci',{}).get('accuracy'))}。"
             "這表示模型仍會偏離專家標籤，不能由單一分數直接推導候選是否符合整體品質要求。")
    report.p("每筆比較都能在 scores.csv 查到兩個候選分數、差值與判斷；predictions.csv 則將各方法攤開，"
             "可按錯誤類別逐筆篩選。不同候選的字詞長度不一致，因此相似度平均值只作文字差距的描述，不用來取代專家一致率。", size=9.5)

    # 7. Every matrix uses identical axes and unweighted row counts.
    report.heading("七  混淆矩陣與錯誤方向", new_page=True)
    report.p("下圖的列為專家答案、欄為指標預測。對角線上的筆數相加即為正確筆數；"
             "最左上與最右下分別是正確選 B 與正確選 A，中間格是正確平手。"
             "左下及右上代表把專家偏好的方向完全反轉。")
    report.figure(figures["confusion"], "圖 2  五項方法的測試集混淆矩陣  每格為比較筆數")
    report.table(["方法", "反向誤判筆數", "占非平手答案比例", "平手召回率"], [
        [metric_label(m, results), test[m]["reverse_count"], pct(test[m]["reverse_rate"]), pct(test[m]["tie_recall"])]
        for m in methods], [5.0, 3.5, 4.0, 3.9], number_columns=(1, 2, 3))
    report.p("平手列可看出方法是否有能力保留兩者相當的答案；兩側勝負列則可看出它偏向過度判平手或選錯方向。"
             "某方法即使非平手資料的排序較好，也可能因平手判斷不足而降低三分類正確率。", size=9.5)

    # 8. Class-specific performance, supporting the interpretation of tie errors.
    report.heading("八  平手判斷與各類別表現", new_page=True)
    report.p("測試集有 49 筆平手，占 26.63%；這 49 筆的 A、B 原始文本都不完全相同。"
             "因此以分數完全相等作為平手條件，並不能假定會重現專家的品質相當判斷。")
    report.figure(figures["ties"], "圖 3  預測平手的總筆數與其中正確的筆數")
    report.table(["方法", "專家類別", "筆數", "Precision", "Recall", "F1"], stat_class_table(results),
                 [4.0, 2.7, 1.5, 3.1, 3.1, 3.0], font_size=8.7, number_columns=(2, 3, 4, 5))
    report.p("Precision 回答某類預測有多少是真的；Recall 回答該類專家答案有多少被抓到。"
             "例如平手 precision 高、recall 低，表示判為平手時通常可靠，但漏掉很多專家認定相當的比較。", size=9.3)

    # 9. Train is explicitly descriptive/in-sample.
    report.heading("九  訓練資料的描述性結果", new_page=True)
    report.p("下列 1,044 筆訓練比較與測試結果分開呈現。PatClaimEval 的訓練過程使用此官方訓練資料，"
             "因此本表屬已見資料上的表現；它不能當成額外的獨立測試證據。"
             "其餘文字相似度方法沒有在本次依訓練標籤重新訓練語言模型。")
    report.table(methods_header, primary_table(results, "train"), widths, number_columns=(1, 2, 3, 4, 5))
    report.table(["方法", "訓練正確率", "測試正確率", "訓練減測試", "訓練 Weighted F1"], [
        [metric_label(m, results), pct(train[m]["accuracy"]), pct(test[m]["accuracy"]),
         f"{(train[m]['accuracy']-test[m]['accuracy'])*100:+.2f} 個百分點", num(train[m]["weighted_f1"])]
        for m in methods], [4.0, 3.0, 3.0, 4.0, 3.4], number_columns=(1, 2, 3, 4))
    report.heading("如何解讀訓練與測試差距", level=2)
    report.p("兩表差距受樣本組成、比較難度及模型是否看過標籤影響。訓練較高通常提示已見資料較容易被重現，"
             "但不能只憑差距判定某一個機制導致過度擬合。測試候選本身全在訓練出現，也使這個差距低估部署到新候選時可能面對的變化。")
    report.p("原始訓練檔含 20 筆完全重複列。本表保留官方原始列，因此重複比較具有原檔指定的權重。"
             "敏感度結果另揭露去除重複之後的變化，以避免讀者把重複列當作不同發明。")
    report.heading("基準值", level=2)
    report.p(f"在測試集，總是預測 B 較佳的正確率為 {pct(majority)}。均勻隨機選三類的期望正確率為 33.33%，"
             "這是理論期望，沒有用隨機抽一次的結果替代。固定預測 B 的基準由訓練集的多數類選定，"
             "再直接用於測試集，沒有依測試答案調整評估方法。")

    # 10. Sensitivity analyses are reported with their own denominator.
    report.heading("十  重複比較與參考複製的敏感度", new_page=True)
    report.p("主結果保留官方切分中的全部紀錄。下列敏感度分析各自改變納入條件，"
             "應在同一子集內比較方法；不同子集正確率的差不能單獨歸因於某個候選特徵。")
    cohort_keys = ["test_no_train_comparison", "test_no_gold_copy", "train_deduplicated"]
    cohort_titles = {"test_no_train_comparison": "測試排除重複比較", "test_no_gold_copy": "測試排除參考複製", "train_deduplicated": "訓練去除完全重複"}
    cohort_headers = ["分析子集", "筆數", "BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore", "PatClaimEval"]
    cohort_rows = [[cohort_titles[key], results["cohorts"][key]["n"]] +
                   [pct(results["cohorts"][key]["results"][m]["accuracy"]) for m in methods] for key in cohort_keys]
    report.p("三分類正確率", bold=True)
    report.table(cohort_headers, cohort_rows, [4.0, 1.2, 2.4, 2.4, 2.4, 2.5, 2.5], font_size=8.4, number_columns=(1, 2, 3, 4, 5, 6))
    report.p("Macro F1", bold=True)
    report.table(cohort_headers, [[cohort_titles[key], results["cohorts"][key]["n"]] +
                  [num(results["cohorts"][key]["results"][m]["macro_f1"], 3) for m in methods] for key in cohort_keys],
                 [4.0, 1.2, 2.4, 2.4, 2.4, 2.5, 2.5], font_size=8.4, number_columns=(1, 2, 3, 4, 5, 6))
    report.p("9 筆跨切分重複的測試列號為 28、39、61、97、101、121、151、158、180。"
             "忽略 A/B 順序後亦只有這 9 筆，未發現其他反向重複。排除此類比較能降低直接重複的影響，"
             "但剩下的候選文本仍全在訓練出現。")
    report.p("測試集有 28 筆包含與 gold_claim 原文完全相同的候選；這些比較的專家都偏好該候選。"
             "相似度方法在此具有直接優勢。排除此類後的 156 筆結果，較接近兩份候選都經改寫的比較情境。")
    report.p("壓縮連續空白後，測試第 97 筆 A、B 會變成相同文字，但專家標籤為 B 較佳；"
             "訓練集也存在此類非平手標註。因此本次保存原文，將 tokenizer 如何對待空白視為方法的一部分，"
             "不以自行改寫原文來消除表面上的標籤困難。", size=9.5)

    # 11. Show actual truncation exposure without conflating word and token counts.
    report.heading("十一  長文本與截斷影響", new_page=True)
    report.figure(figures["length"], "圖 4  完整文本詞數與 PatClaimEval 預設長度的截斷範圍")
    report.p("左圖的詞數是依空白切分的描述性長度，與模型 tokenizer 的 token 數不同。"
             "PatClaimEval 將 gold 與候選合併後計算 token；其 2,048 上限不是每段各 2,048。")
    long_rows = [[row["row_number"], candidate.upper(), row[f"patclaim_{candidate}_tokens"], LABEL_NAMES[row["human_label"]]]
                 for row in test_comparisons for candidate in ["a", "b"] if row[f"patclaim_{candidate}_truncated"]]
    report.table(["測試列號", "受影響候選", "未截斷配對 token", "專家答案"], long_rows,
                 [3.2, 4.0, 6.2, 4.0], number_columns=(0, 1, 2, 3))
    report.p("預設長度在測試集影響 3 / 184 筆比較，以及 3 / 368 個候選評分。訓練集影響 12 筆比較，"
             "涉及 12 / 2,088 個候選評分。上表兩個不同的超長候選在多筆比較中重複出現。")
    if "patclaimeval_4096" in test:
        full = test["patclaimeval_4096"]
        report.table(["PatClaimEval 設定", "正確筆數", "正確率", "Macro F1"], [
            ["官方預設 2048", pat["correct"], pct(pat["accuracy"]), num(pat["macro_f1"])],
            ["補充長度 4096", full["correct"], pct(full["accuracy"]), num(full["macro_f1"])],
        ], [5.8, 3.2, 4.2, 4.2], number_columns=(1, 2, 3))
        report.p(f"4096 設定的整體正確率較 2048 改變 {(full['accuracy']-pat['accuracy'])*100:+.2f} 個百分點。"
                 "這項成對比較同時改變輸入可見範圍及固定 padding 長度；所有候選都各跑兩種設定，未重新訓練模型。", size=9.3)
    bert_audit = results["audit"]["truncation"]["test"]
    bert_train_audit = results["audit"]["truncation"]["train"]
    report.p(f"BERTScore 在測試集有 {bert_audit['bertscore_rows']} / 184 筆比較至少一段文本超過 512 tokens。"
             f"受影響候選評分為 {bert_audit['bertscore_pair_instances']} / 368；訓練集則為 "
             f"{bert_train_audit['bertscore_rows']} / 1,044 筆比較、{bert_train_audit['bertscore_pair_instances']} / 2,088 個候選評分。"
             "其超長與未超長子集的結果如下；兩群文字內容不同，差異屬描述統計，不能當成截斷的因果效果。", size=9.3)
    report.table(["BERTScore 測試子集", "筆數", "正確率", "Macro F1"], [
        [title, results["cohorts"][key]["n"], pct(results["cohorts"][key]["results"]["bertscore_f1"]["accuracy"]),
         num(results["cohorts"][key]["results"]["bertscore_f1"]["macro_f1"])]
        for key, title in [("test_bertscore_truncated", "至少一段超過 512"), ("test_bertscore_untruncated", "所有文本均未超過 512")]],
        [6.4, 2.5, 4.2, 4.3], font_size=8.5, number_columns=(1, 2, 3))

    # 12. Same-sample paired intervals and cautious historical comparison.
    report.heading("十二  方法差異與原論文比較", new_page=True)
    paired = [item for item in results.get("paired_differences", [])
              if item.get("cohort") in ["test", "official_test", "test_all"]
              and (item["method_a"] == "patclaimeval" or item["method_b"] == "patclaimeval")]
    if not paired:
        paired = [item for item in results.get("paired_differences", []) if "patclaimeval" in [item["method_a"], item["method_b"]]][:5]
    report.table(["方法 A 減方法 B", "正確率差", "95% 成對信賴區間"], [
        [metric_label(item["method_a"], results) + " − " + metric_label(item["method_b"], results),
         f"{item['delta_accuracy']*100:+.2f} pp",
         f"{item['ci_accuracy'][0]*100:+.2f} 至 {item['ci_accuracy'][1]*100:+.2f} pp"] for item in paired],
        [8.7, 3.3, 5.4], number_columns=(1, 2))
    report.p("pp 表示百分點。若成對區間包含 0，這批資料尚無法清楚區分兩方法的正確率；"
             "若區間完全大於 0，支持方法 A 在此樣本設定下較高；完全小於 0 則支持方法 B 較高。此處為描述性多組比較，未另調整多重比較，"
             "不能把區間結果擴大為對其他資料來源的保證。Macro F1 的成對差值及區間亦保存於 results.json 的 paired_differences。")
    paper_rows = []
    for item in results.get("paper_comparison", []):
        method = item["method"]
        paper_accuracy = item["accuracy"]
        if paper_accuracy > 1:
            paper_accuracy /= 100
        paper_f1 = item.get("reported_f1")
        if paper_f1 is not None and paper_f1 > 1:
            paper_f1 /= 100
        paper_rows.append([metric_label(method, results), pct(paper_accuracy, 1), pct(test[method]["accuracy"]),
                           f"{(test[method]['accuracy']-paper_accuracy)*100:+.2f}", num(paper_f1, 3),
                           num(test[method]["weighted_f1"], 3)])
    report.table(["方法", "論文 Acc", "本次 Acc", "差值 pp", "論文 F1", "本次加權 F1"], paper_rows,
                 [3.4, 2.4, 2.5, 2.5, 2.6, 3.0], font_size=8.7, number_columns=(1, 2, 3, 4, 5))
    report.p("原論文整體品質表格回報 PatClaimEval accuracy 為 56.5%、F1 為 57.4%、Kendall τ 為 0.477、"
             "Spearman ρ 為 0.602。原表未明示 F1 的平均方式，也未完整說明相關係數輸入為連續分數差或離散預測；"
             "因此本表的 F1 僅並列參考，不能視為精確重現差值。 [1]")
    report.p("論文正確率僅顯示至百分比小數一位，因此極小的百分點差異也包含顯示四捨五入的影響；"
             "應先比較是否落在相同的顯示精度，再討論真正的逐筆預測差異。", size=9.3)
    report.p("本次使用公開固定 checkpoint 與記錄完整的分詞、長度及決策設定。若與論文數字不同，"
             "應先查核模型版本、BLEU/ROUGE 實作、BERTScore 模型設定、平手規則與截斷，而非直接推論原研究錯誤或本次模型改善。")
    report.table(["方法", "連續 Δ 的 τ", "連續 Δ 的 ρ", "離散預測的 τ", "離散預測的 ρ"], [
        [metric_label(m, results), num(test[m]["continuous_tau"], 3), num(test[m]["continuous_rho"], 3),
         num(test[m]["discrete_tau"], 3), num(test[m]["discrete_rho"], 3)] for m in methods],
        [4.3, 3.2, 3.2, 3.3, 3.4], font_size=8.6, number_columns=(1, 2, 3, 4))

    # 13. Explain the gap to the reference in its native numerical scale.
    report.heading("十三  候選與參考文本的分數差距", new_page=True)
    report.p("本節描述候選相對於 gold_claim 的分數分布。A 與 B 只是每筆資料的候選位置，"
             "不是兩個已知生成模型；公開資料沒有逐列生成模型名稱，因此不能把 A 的平均分較高解讀為某模型勝出。")
    report.table(["方法", "A 平均", "B 平均", "候選平均", "候選中位數", "候選 Q1 至 Q3", "Δ 平均"], [
        [metric_label(m, results), num(test[m]["score_statistics"]["A"]["mean"]), num(test[m]["score_statistics"]["B"]["mean"]),
         num(test[m]["score_statistics"]["pooled_candidates"]["mean"]),
         num(test[m]["score_statistics"]["pooled_candidates"]["median"]),
         num(test[m]["score_statistics"]["pooled_candidates"]["q1"]) + " 至 " + num(test[m]["score_statistics"]["pooled_candidates"]["q3"]),
         num(test[m]["score_statistics"]["delta"]["mean"])] for m in methods],
        [3.4, 1.8, 1.8, 2.0, 2.1, 4.3, 2.0], font_size=8.4, number_columns=(1, 2, 3, 4, 5, 6))
    report.p("候選分布合併每列的 A 與 B，共 368 個候選出現次數；同一候選若在多列出現，就依原始列數重複計入。"
             "這是依比較資料權重計算的分布，未去重成 186 個不同參考候選配對。Q1 至 Q3 為中間 50% 的分數範圍。", size=9.4)
    report.p("文字相似度中的分數不足 1，可用來定位改寫、刪減、重排或新增文本；不同方法對這些變化的敏感程度不同。"
             "PatClaimEval 的 sigmoid 分數則沒有相同的『距離 1 等於錯誤比例』解釋，也不能用 1 − s 表示遺漏特徵的比例。")
    report.heading("分數與答案不一致時如何檢查", level=2)
    report.p("先核對輸入是否為同一 gold 下的 A/B，再確認指標有沒有截斷、差值是否落在平手區間，以及專家答案是否為平手。"
             "接著檢視請求項的必要技術特徵、主體名稱、依附關係及限定條件。這些文字檢查能提出可能原因，"
             "公開資料未提供逐筆專家評語，因此不能反推為專家的既定理由。")
    report.heading("結果可支持的用途", level=2)
    report.p("這批結果可用來挑選與專家偏好較一致的研究評估器、篩選值得人工查看的候選差異，"
             "以及建立可重現的品質比較流程。若後續目標是處理新專利，需另外建立以參考文本或專利來源分組的獨立測試集，"
             "避免相同候選在訓練與測試同時出現。")
    report.p("若要將品質評估細分為特徵完整性、概念清晰度、術語一致性與邏輯連結，需另取得這些面向的人工標註及對應模型。"
             "本次 Quality 的偏好正確率，無法當成這四個面向各自的正確率。")

    # 14--17. Eight literal, auditable examples. No invented expert rationale.
    cases = results.get("cases", [])
    for page_index in range(0, len(cases), 2):
        number = 14 + page_index // 2
        report.heading(f"{['十四','十五','十六','十七'][page_index//2]}  原文案例分析 {page_index+1} 至 {min(page_index+2,len(cases))}", new_page=True)
        if page_index == 0:
            report.p("案例依錯誤型態、專家平手、截斷及資料重複等條件選取，用於解釋差異，並非隨機抽樣；"
                     "成效以完整測試集判讀。原文分析為本報告觀察，逐筆專家理由未公開。", size=9.0, after=5)
        for case_index, case in enumerate(cases[page_index:page_index+2], start=page_index+1):
            report.heading(f"案例 {case_index}  測試第 {case['row_number']} 筆", level=2)
            selected = case.get("reason_selected", "代表性比較")
            if isinstance(selected, list):
                selected = "、".join(selected)
            report.p(f"專家答案：{LABEL_NAMES[case['human_label']]}。選例依據：{selected}。", size=9.2, after=4)
            a_excerpt, b_excerpt, location = differing_excerpts(case["A"], case["B"])
            report.p("參考原文節錄  " + excerpt(case["gold_claim"], 120), size=8.7, after=3)
            report.p("候選節錄位置  " + location + "。原文僅截取顯示，完整文字見逐筆檔。", size=8.3, after=3)
            report.p("候選 A 原文節錄  " + a_excerpt, size=8.7, after=3)
            report.p("候選 B 原文節錄  " + b_excerpt, size=8.7, after=3)
            report.table(["方法", "A 分數", "B 分數", "A 減 B", "預測", "一致"], [
                [metric_label(m, results), num(case["methods"][m]["score_a"], 6), num(case["methods"][m]["score_b"], 6),
                 num(case["methods"][m]["delta"], 8), LABEL_NAMES[case["methods"][m]["prediction"]],
                 "是" if case["methods"][m]["correct"] else "否"] for m in methods],
                [4.1, 2.5, 2.5, 2.5, 3.1, 1.7], font_size=8.1, number_columns=(1, 2, 3, 4, 5))
            if case.get("text_observation"):
                report.p("原文內容分析  " + case["text_observation"], size=9.0, after=5)
            truncation_notes = []
            if case.get("patclaim_truncated"):
                truncation_notes.append("PatClaimEval 2048")
            if case.get("bertscore_truncated"):
                truncation_notes.append("BERTScore")
            if truncation_notes:
                report.p("截斷註記  本筆受影響設定為 " + "、".join(truncation_notes) + "。", size=8.5, after=4)

    # 18--19. Executable reproduction instructions, then primary sources.
    report.heading("十八  重現設定與執行方式", new_page=True)
    report.p("所有評分均以固定原始資料與模型設定產生，原始 JSON 沒有改寫。"
             "分數以未四捨五入值進行預測與統計；報告表格才作顯示用四捨五入。"
             "完整設定、執行環境、輸入雜湊及驗證結果存於 results.json 與各評分工作檔。")
    report.table(["產物", "用途"], [
        ["results.json", "整體與子集統計、信賴區間、案例與驗證資訊。"],
        ["scores.csv", "1,228 筆原始比較及六種設定的候選分數與判斷。"],
        ["predictions.csv", "逐方法攤開的預測、答案及是否一致。"],
        ["similarity_scores.json", "BLEU、ROUGE、BERTScore 分數與設定。"],
        ["patclaim_scores.json", "PatClaimEval 2048 與 4096 分數與長度資訊。"],
        ["pairs.json", "輸入文本識別碼、原始切分與資料來源紀錄。"],
    ], [5.0, 12.4], font_size=8.9)
    report.heading("在本機完整重跑", level=2)
    report.p("從專案根目錄的 PowerShell 執行下列命令。新資料夾名稱可自行指定；若中斷後要續跑，"
             "使用相同輸出目錄與相同參數，工作程式會驗證 checkpoint 的輸入與設定後接續。", size=9.3)
    report.p("Set-Location -LiteralPath 'C:\\Users\\agent\\Project\\專利挑戰'", size=8.3, after=4)
    report.p("& '.\\LLM4DPCG\\.venv\\Scripts\\python.exe' '.\\public-programs\\verification\\run_patent_ce_evaluation.py' "
             "--stage all --output-dir '.\\outputs\\patent_ce_evaluation_reproduction' --threads 4 --batch-size 4 --bootstrap-replicates 10000", size=8.3, after=6)
    report.p("協調程式依序執行 prepare、similarity、patclaim、aggregate、verify、report，最後匯出 DOCX、PDF 及供版面查核的 PNG。"
             "PatClaimEval 使用已驗證的本機 WSL Python 與本機模型；相似度工作程式使用專案 Windows Python。"
             "所有模型評分均離線執行。DOCX 建立則使用 Codex bundled Python。", size=9.3)
    report.table(["固定項目", "實際設定"], [
        ["文字前處理", "無自訂正規化，保留來源字串；套件內部行為另記錄。"],
        ["BLEU 及 ROUGE", "BLEU order 4 不平滑；ROUGE 不用 stemming，回報逐文本 F1。"],
        ["BERTScore", "roberta-large 第17層，無 IDF、無 baseline rescale，512 tokens。"],
        ["PatClaimEval", "eval 模式，float32，固定 padding 2048；補充設定 4096。"],
        ["預測規則", "|A分數−B分數| < 0.0001 才判平手，邊界保留勝負。"],
        ["不確定性", "20組 gold 群組重抽樣 10000次，seed 42，方法共用抽樣。"],
    ], [4.0, 13.4], font_size=8.8)
    report.p("逐筆驗證會重新計算分數差、預測標籤、混淆矩陣與正確率，核對資料列與各方法的覆蓋，"
             "並確認原始資料 SHA-256。驗證狀態存於 results.json 的 verification。", size=9.3)
    report.heading("十九  固定版本與主要來源", new_page=True)
    report.p("以下版本與雜湊對應本次實際讀取的來源。以相同文字、套件、模型、長度、固定平手規則重跑，"
             "才能區分資料或設定變動帶來的差異。", size=9.5)
    report.p("資料版本  8a9df74271be1330cf3b185c550e82513cf265f6", size=8.6, after=3)
    report.p("PatClaimEval checkpoint  c959e676f7eb61e32b0f8d9c4cc7d72a4aeb94d5", size=8.6, after=3)
    report.p("訓練 JSON SHA-256  20c3e88862b0a8cd3ce6e48607f9887a29c4502d4e87633b41095be194bdca48", size=8.0, after=3)
    report.p("測試 JSON SHA-256  10e68f47f2f4329f92c6d60936d23f4eaab6b852150b8fa1b039d8d348868039", size=8.0, after=5)
    report.p("BERTScore roberta-large revision  722cf37b1afa9454edce342e7895e588b6ff1d59", size=8.6, after=5)
    report.heading("主要來源", level=2)
    report.link("[1] ACL 2025 論文與正式 PDF", "https://aclanthology.org/2025.acl-long.190/",
                "Jiang, Lekang；Scherz, Pascal A.；Goetz, Stefan（2025）。Towards Better Evaluation for Generated Patent Claims。ACL，3775–3788。DOI 10.18653/v1/2025.acl-long.190。")
    report.link("[2] PatClaimEval 官方程式庫與公開資料", "https://github.com/scylj1/PatClaimEval/tree/8a9df74271be1330cf3b185c550e82513cf265f6",
                "本次固定版本的 README 與 quality_train/test。作者標示授權為 CC BY-NC 4.0。")
    report.link("[3] BLEU 原始論文", "https://aclanthology.org/P02-1040/", "Papineni 等人（2002）。BLEU a Method for Automatic Evaluation of Machine Translation。")
    report.link("[4] ROUGE 原始論文", "https://aclanthology.org/W04-1013/", "Lin（2004）。ROUGE a Package for Automatic Evaluation of Summaries。")
    report.link("[5] BERTScore 官方實作", "https://github.com/Tiiiger/bert_score", "Zhang 等人。BERTScore Evaluating Text Generation with BERT。模型與套件設定影響分數，應依本次 metadata 重現。")
    report.p("報告生成  如評分與統計已完成，可使用 bundled Python 執行 build_report.py --results <results.json> "
             "--output-dir <產出目錄>。圖表直接取用同一份統計結果；DOCX 透過隔離 Word COM 匯出 PDF，"
             "再由標準文件 renderer 與 Poppler 產生每頁 PNG 供版面查核。", size=9.0)
    path = report.save()
    (output_dir / "report_build_manifest.json").write_text(json.dumps({
        "report": str(path), "sha256": sha256(path), "primary_methods": methods,
        "results_source": "results.json", "figures": {key: str(value) for key, value in figures.items()},
        "planned_pages": 19, "cases": len(cases), "render_required": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = read_json(args.results)
    validate_report_inputs(results, args.results.parent)
    path = build_report(results, args.output_dir, args.results.parent)
    print(json.dumps({"docx": str(path), "size": path.stat().st_size, "planned_pages": 19}, ensure_ascii=False))


if __name__ == "__main__":
    main()
