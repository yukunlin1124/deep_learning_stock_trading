"""Generate the full course final report as report.docx.

First-draft generator — produces a ~20 page Traditional Chinese report
covering all sections of Outline X. User edits in Word from here.

Embeds: 3 matplotlib figures + 2 native diagrams + 1 hyperparameter table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Match a Greek letter optionally followed by ASCII letters / digits / more
# Greek letters — captures Greek-start identifiers like "α" or "θ" while not
# gobbling Mandarin text. Note: limit italics to Greek-starting tokens so
# function names like MSE or words like "epoch" remain upright.
_ITALIC_PAT = re.compile(r"[α-ωΑ-Ω][a-zA-Z0-9α-ωΑ-Ω]*")

# Match Identifier_subscript patterns where the base is a single Latin /
# Greek / math letter (optionally followed by a combining diacritic like
# x̃) and the subscript is alphanumeric. Captures the underscored form
# (L_train, η_ψ, F_θ, y_raw, x_test, x̃_test, ∇_φ, ...) so we can render
# the subscript portion via python-docx's run.font.subscript = True.
_SUB_PAT = re.compile(
    # base: Latin/Greek/∇ letter + optional combining diacritic + optional
    # Unicode superscript chars (e.g. ᵏ, ⁽ᵏ⁻¹⁾, ², …) so patterns like
    # Dᵏ_train, F̃ᵏ_test, D̃⁽ᵏ⁻¹⁾_train match cleanly.
    r"([A-Za-zα-ωΑ-Ω∇][̀-ͯ]?[²-¹⁰-₟ᴬ-ᵪ]*)"
    r"_([A-Za-zα-ωΑ-Ω0-9]+)"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "docs" / "report.docx"
FIG = PROJECT_ROOT / "docs" / "figures"
OUTPUT_BASE = PROJECT_ROOT / "models"

CJK_FONT = "標楷體"
EN_FONT = "Times New Roman"


def set_cjk(run, size=11, bold=False, italic=False, color=None):
    """Set both CJK and Latin fonts on a run, ACM-paper style."""
    run.font.name = EN_FONT
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), CJK_FONT)
    rFonts.set(qn("w:ascii"), EN_FONT)
    rFonts.set(qn("w:hAnsi"), EN_FONT)
    rFonts.set(qn("w:cs"), EN_FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    # ACM paper style: all body and section text is black; emphasis via bold only.
    run.font.color.rgb = color or RGBColor(0x00, 0x00, 0x00)


def add_heading(doc, text, level=1, color=None):
    """KDD/ACM-style section heads: bold black, larger sizes for level 1.

    Level 1 (e.g. "1 前言"): 12pt bold, all-caps-ish via size only.
    Level 2 (e.g. "2.1 ..."): 11pt bold.
    Level 3 (e.g. "2.1.1 ..."): 10.5pt bold italic.
    """
    h = doc.add_paragraph()
    if level == 1:
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        set_cjk(r, size=12, bold=True, color=RGBColor(0, 0, 0))
    elif level == 2:
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        set_cjk(r, size=11, bold=True, color=RGBColor(0, 0, 0))
    else:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
        r = h.add_run(text)
        set_cjk(r, size=10.5, bold=True, color=RGBColor(0, 0, 0))


def _emit_italic_segment(paragraph, text, size, bold, color=None):
    """Inner: italicize Greek-start tokens, leave the rest upright."""
    pos = 0
    for m in _ITALIC_PAT.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos : m.start()])
            set_cjk(r, size=size, bold=bold, color=color)
        r = paragraph.add_run(m.group())
        set_cjk(r, size=size, bold=bold, italic=True, color=color)
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        set_cjk(r, size=size, bold=bold, color=color)


def _add_runs_italic_greek(paragraph, text, size, bold, color=None):
    """Add text to a paragraph with two visual transforms:
    1. Identifier_subscript pairs (L_train, η_ψ, F_θ) render the part after
       the underscore as italic + subscript, NO literal underscore shown.
    2. Greek-start tokens render italic; ASCII / Mandarin stay upright."""
    pos = 0
    for m in _SUB_PAT.finditer(text):
        if m.start() > pos:
            _emit_italic_segment(paragraph, text[pos : m.start()], size, bold, color)
        base, sub = m.group(1), m.group(2)
        r = paragraph.add_run(base)
        set_cjk(r, size=size, bold=bold, italic=True, color=color)
        r = paragraph.add_run(sub)
        set_cjk(r, size=size, bold=bold, italic=True, color=color)
        r.font.subscript = True
        pos = m.end()
    if pos < len(text):
        _emit_italic_segment(paragraph, text[pos:], size, bold, color)


def add_para(doc, text, size=10, bold=False, justify=True, indent_cm=0.74):
    """ACM body text: 10pt (Mandarin), justified, single line spacing,
    first-line indent. Greek-letter identifiers auto-italicized."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(indent_cm) if indent_cm else None
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _add_runs_italic_greek(p, text, size=size, bold=bold)
    return p


def add_caption(doc, text):
    """Figure/table caption: 9pt italic, centered, black (no color)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    set_cjk(r, size=9, italic=True, color=RGBColor(0, 0, 0))


def add_image(doc, path, width_cm=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    r = p.add_run()
    r.add_picture(str(path), width=Cm(width_cm))


def add_table(
    doc, headers, rows, col_widths_cm=None, header_color="E8E8E8", style="Light Grid", bold_rows=None, bold_cells=None
):
    """ACM-paper-style table. bold_rows: list of data-row indices to bold
    whole row. bold_cells: list/set of (row_index, col_index) tuples for
    per-cell bold (overrides bold_rows for those cells)."""
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = style
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    bold_set = set(bold_rows or [])
    bold_cell_set = set(bold_cells or [])

    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_runs_italic_greek(p, h, size=9.5, bold=True, color=RGBColor(0, 0, 0))
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), header_color)
        tc_pr.append(shd)

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.rows[i].cells[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            is_bold = (i - 1) in bold_set or (i - 1, j) in bold_cell_set
            _add_runs_italic_greek(p, str(val), size=9, bold=is_bold, color=RGBColor(0, 0, 0))

    if col_widths_cm:
        for j, w in enumerate(col_widths_cm):
            for row in tbl.rows:
                row.cells[j].width = Cm(w)
    add_para(doc, "", indent_cm=0)  # spacer
    return tbl


def add_page_break(doc):
    doc.add_page_break()


def load_summary(run):
    p = OUTPUT_BASE / run / "report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r


def _bt(run, key, default="—"):
    s = load_summary(run) or {}
    bt = s.get("backtest", {})
    v = bt.get(key, default)
    return v


def _pct(v):
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return str(v)


def build_report():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.size = Pt(10)
    style.font.name = EN_FONT
    rPr = style.element.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), CJK_FONT)
    rFonts.set(qn("w:ascii"), EN_FONT)
    rFonts.set(qn("w:hAnsi"), EN_FONT)
    rPr.append(rFonts)

    # ─────────────────────────────────────────────────────────────
    # 封面 / 標題 (ACM/KDD style: black bold title, no color)
    # ─────────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(80)
    title.paragraph_format.space_after = Pt(12)
    r = title.add_run("基於 DoubleAdapt 元學習框架\n之台股自動化交易系統")
    set_cjk(r, size=18, bold=True, color=RGBColor(0, 0, 0))

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(60)
    r = sub.add_run("深度學習 ── 期末專題報告")
    set_cjk(r, size=12, color=RGBColor(0, 0, 0))

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(6)
    r = author.add_run("姓名：林煜崐")
    set_cjk(r, size=12)

    sid = doc.add_paragraph()
    sid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sid.paragraph_format.space_after = Pt(30)
    r = sid.add_run("學號：N16144733")
    set_cjk(r, size=12)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_after = Pt(6)
    r = info.add_run("撰寫日期：2026 年 6 月 26 日")
    set_cjk(r, size=11)

    info2 = doc.add_paragraph()
    info2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info2.paragraph_format.space_after = Pt(40)
    r = info2.add_run("交易期間：2026 年 5 月 ~ 6 月 26 日")
    set_cjk(r, size=11)

    add_para(doc, "", indent_cm=0)
    add_page_break(doc)

    # ─────────────────────────────────────────────────────────────
    # §1 摘要
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "1. 摘要", level=1)
    add_para(
        doc,
        "本專題在課堂股票模擬系統上實作並系統性評估 DoubleAdapt "
        "── 一種基於元學習的「資料適配 + 模型適配」雙適配框架 ── "
        "用於股票報酬預測與每日自動下單。本專題的重點在於：透過受控消融實驗"
        "釐清 DoubleAdapt 各設計元件在台股 TW50 宇宙的具體貢獻，並以一條"
        "完全移除雙適配器的增量學習基線量化兩個適配器的真實效果。"
        "實際部屬組態於 2024-01 至 "
        "2026-04 的回測獲得累積報酬 +56.9%、夏普指數 (Sharp) 1.24、最大"
        "回撤僅 −14.07%（為所有實驗中最低）；無適配器之增量學習基線同視窗"
        "得 +49.96%、夏普指數 0.92、回撤 −26.87%。據此量化兩個適配器貢獻："
        "+0.32 夏普指數與最大回撤減半（−12.8 %），主要價值為下行"
        "保護。線上實際部署至 2026-06-26 為止累積報酬 +40%，排行榜第 8 名。",
    )
    # ─────────────────────────────────────────────────────────────
    # §2 前言
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "2. 前言", level=1)
    add_para(
        doc,
        "量化交易的核心問題在於模型必須在「分布漂移」（Distribution shift）的"
        "環境下持續預測。股市並非穩態系統，今年管用的特徵明年可能完全失效，"
        "因此單純地以歷史資料訓練、長期凍結部署，無法達成穩健的長期收益。"
        "傳統做法 ── Rolling Retraining（RR）每隔一段時間以全歷史資料從零"
        "重訓，雖能緩解漂移但計算成本高昂，且常忽略最新一段尚未驗證的訊號；"
        "另一種做法增量學習（Incremental Learning, IL）只用最新資料微調，效率"
        "高但對分布漂移敏感，過去增量資料中的偏差可能被放大。",
    )
    add_para(
        doc,
        "DoubleAdapt [1] 於 2023 年 KDD 提出，將每個增量學習任務視為一個元學習"
        "任務，透過雙層最佳化同時優化兩個適配器：特徵適配器（G）與標籤"
        "適配器（H），分別對應特徵與標籤的局部分布調整，並讓 GRU 預測基底（F）"
        "能快速適應新區間。原論文於中國 A 股 CSI300/CSI500 進行實驗，本專題的"
        "核心問題為：",
    )
    add_para(
        doc,
        "（i）DoubleAdapt 在較小、更集中的台股 TW50 宇宙上是否同樣有效？"
        "（ii）原論文中每一個關鍵設計（截面排序標準化 Cross-Sectional Rank Normalization 包裝、"
        "最佳狀態還原 best state restore、"
        "FOMAML 任務步長 r、資料適配器學習率）在台股場景貢獻多少？"
        "（iii）相較於不含適配器的增量學習基線，DoubleAdapt 的兩個適配器"
        "具體貢獻多少？",
    )
    add_para(
        doc,
        "本報告以工程演化的視角，從 qlib 預設組態出發，逐步消除問題、找到"
        "最佳實際部屬組態，並以完全移除雙適配器的增量學習基線驗證雙適配器"
        "的真實貢獻。最後將實際部屬模型部署至課堂交易系統，截至 "
        "2026-06-26 為止累積報酬 +40%，排行榜第 8 名。",
    )

    # ─────────────────────────────────────────────────────────────
    # §3 相關工作
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "3. 相關工作", level=1)
    add_para(
        doc,
        "DoubleAdapt 的方法論直接奠基於 Finn 等人 [2] 提出的 Model-Agnostic "
        "Meta-Learning（MAML）。MAML 將「快速適應」公式化為雙層最佳化："
        "下層在支持集（meta-train task）做一步梯度更新得到任務專屬參數 θ；"
        "上層在查詢集（meta-test task）以 θ 計算損失並回傳更新到元學習器的"
        "初始參數 φ。本實作採用 first-order MAML（FOMAML）變體 ── 內層"
        "梯度視為常數，省去二階導數的記憶體與計算成本。",
    )
    add_para(
        doc,
        "Microsoft 的 qlib [3] 為本實作提供了：Alpha360 60×6 特徵格式、"
        "Top-K Dropout 投組規則、IC / ICIR 標準化評估指標、與整套回測引擎"
        "慣例。我們直接引用 qlib 對 CSI300 上「top-K=50、每日替換 5 檔」"
        "的設計準則（10% 日換手率），按台股宇宙較小的特性調整為 "
        "「top-K=10、每日替換 1 檔」（仍維持 10% 換手率）。",
    )
    add_para(
        doc,
        "DoubleAdapt 原論文 [1] 在 CSI300/CSI500 上比較了滾動再訓練、增量"
        "學習、與 DoubleAdapt 三類做法，顯示 DoubleAdapt 在風險調整指標上"
        "優於增量學習且訓練成本遠低於滾動再訓練。本專題的相關工作是將該"
        "方法套用至台股 TW50 宇宙，並以受控消融實驗量化各設計元件在台股"
        "場景的貢獻。",
    )

    # ─────────────────────────────────────────────────────────────
    # §4 系統設計
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "4. 系統設計", level=1)

    add_heading(doc, "4.1 資料來源", level=2)
    add_para(
        doc,
        "資料來源為台灣經濟新報（TEJ）的半年度個股還原日線資料，"
        "涵蓋 2016-01 至 2026-06 共 10.5 年的開盤、最高、最低、收盤、"
        "成交量、成交金額。相對於免費資料源不一致的除權除息調整，TEJ 對"
        "全歷史套用統一的還原（split + dividend back-adjustment），使預測"
        "標籤（次日收盤報酬）無人為跳空，是專題前期切換資料源後獲得 "
        "夏普指數 +0.10、最大回撤減半的關鍵原因。",
    )
    add_para(
        doc,
        "宇宙選擇為台灣 50 指數（TW50，約 50 檔大型權值股），依 TEJ 提供的"
        "官方月度成分快照（2011-2026 共 184 筆更新）動態套用，反映 TW50 "
        "半年度的成分輪替。歷史視窗長度為 60 個交易日（每筆樣本由過去 "
        "60 日特徵組成），預測標的為次日收盤報酬。經此過濾後，有效候選池"
        "共約 99 個唯一代號。",
    )

    add_heading(doc, "4.2 特徵工程", level=2)
    add_para(
        doc,
        "每筆訓練樣本由「6 個欄位 × 60 日歷史」組成共 360 維特徵，"
        "完全對應 qlib 的 Alpha360 規格。6 欄位依序為收盤、開盤、最高、"
        "最低、成交量加權均價（VWAP）、成交量，其中 VWAP 由成交金額除以"
        "成交量計算（成交量為零時退回收盤價）。標準化採 qlib 慣例的"
        "「per-field」模式 ── 每欄位在每個樣本內以最新 (lag-0) 之值為分母"
        "歸一化，使所有特徵在尺度上可比。標籤為次日收盤報酬，遵循 "
        "DoubleAdapt 原論文的股票價格趨勢定義。",
    )
    add_para(doc, "資料層的主要關鍵符號與計算公式整理如下表：")
    add_table(
        doc,
        headers=["符號", "中文意義", "公式或值", "備註"],
        rows=[
            ["X(t)", "特徵矩陣", "形狀 = (B, 60, 6)", "每筆樣本為過去 60 日的 6 欄位"],
            ["y(t)", "次日收盤報酬", "y(t) = close(t+1) / close(t) − 1", "預測標的"],
            ["VWAP(t)", "成交量加權均價", "VWAP(t) = turnover(t) / volume(t)", "若成交量為 0 則退回收盤價"],
            ["X̃(t)", "歸一化特徵", "X̃(t) = X(t) / X(t)[lag-0]", "每欄位獨立歸一化"],
            ["L_train", "內層 MSE 損失", "MSE(F(G(X_train)), H(y_train))", "對 G/H 變換後的標籤"],
            ["L_test", "外層複合損失", "MSE(ŷ, y_test) + 0.5 · MSE(H(y), y_test)", "0.5 為正則項權重 α"],
            ["Sharp", "夏普指數", "mean(R) / std(R) · √252", "風險調整年化報酬"],
        ],
        col_widths_cm=[2.0, 2.6, 6.6, 4.8],
        style="Table Grid",
    )
    add_para(
        doc,
        "為穩定訓練，特徵在預訓練資料上以 穩健 Z-Score 標準化 (Robust Z-Score Normalization) 估計每 "
        "(歷史時間步, 欄位) 對的中位數與絕對中位差（MAD），並套用於整個面板。"
        "標籤則於每個任務內做「截面排序標準化 (Cross-Sectional Rank Normalization)」── 此設計為 qlib "
        "既有穩定性技巧，將每日的標籤值轉為標準常態分布上的排名分數，"
        "避免極端報酬主宰梯度。本專題的消融實驗（將其移除的對照組）顯示"
        "移除它會使夏普指數減半，意指此標準化對訓練動力學至關重要。",
    )

    add_heading(doc, "4.3 模型架構", level=2)
    add_para(
        doc,
        "模型由三個元件組成：特徵適配器（G）、GRU 基底預測器（F）、"
        "與標籤適配器（H），對應原論文的「資料適配器 DA」與「模型適配器 MA」。"
        "各元件實作直接對齊原論文於 qlib 中公開的參考程式：",
    )
    add_image(doc, PROJECT_ROOT / "docs" / "figure1.png", width_cm=15.5)
    add_caption(doc, "圖 1：DoubleAdapt 架構 ── G + F + H 三元件流向")
    add_para(
        doc,
        "如圖 1 所示，整個架構可分為四個概念性步驟，以下依序說明。每個步驟" "對應原論文 Algorithm 1 中的一個明確環節。",
    )

    add_para(doc, "(1) 增量資料調適 (Incremental data adaptation)。", bold=True, indent_cm=0)
    add_para(
        doc,
        "給定第 k 個任務的增量資料 Dᵏ_train，資料適配器（DA）將每個樣本的"
        "特徵向量 x 透過特徵適配器 G 進行轉換，並將對應的標籤 y 透過標籤"
        "適配器 H 進行轉換，產生一份「已調適」的增量資料集 D̃ᵏ_train。"
        "在本實作中，G 對每個時間步以餘弦 (cosine) 相似度匹配 N = 8 個學習"
        "得到的「原型」向量，經歸一化指數函數 (softmax) 取得每個預測頭 "
        "(head) 的權重，再以多頭殘差變換得到 x̃ = x + Σᵢ sᵢ · headᵢ(x)；"
        "H 對 x̃ 計算獨立門控值，以每個預測頭的仿射 hᵢ(y) = γᵢ·y + βᵢ "
        "變換標籤，其中 γ 初值取 U(0.75, 1.25)、β = 1/8。",
    )

    add_para(doc, "(2) 模型適應 (Model adaptation)。", bold=True, indent_cm=0)
    add_para(
        doc,
        "模型適配器（MA）以參數權重 φ⁽ᵏ⁻¹⁾ 初始化預測模型；接著依循一般"
        "增量學習做法，在已調適的 D̃ᵏ_train 上微調預測模型，產生任務專屬"
        "參數 θᵏ。最後將更新後的預測模型 F(·; θᵏ) 部署上線。F 結構為輸入"
        "維度 6、隱藏維度 64、共 2 層的標準 GRU，後接全連接層輸出純量預測，"
        "與 qlib 的 GRU 預設完全一致；F 共有 38,849 個參數，即 MA 部分。",
    )

    add_para(doc, "(3) 線上推論 (Online inference)。", bold=True, indent_cm=0)
    add_para(
        doc,
        "給定測試集 Dᵏ_test 中每個樣本的特徵 x，DA 先透過 G 將其轉換為 "
        "x̃；接著 F(·; θᵏ) 以 x̃ 為輸入產生中介預測 y̌，y̌ 再透過 H⁻¹ 轉回"
        "原始報酬尺度，得到最終預測 ŷ。轉換 Dᵏ_test 中的原始特徵後可"
        "獲得已調適的測試資料集 D̃ᵏ_test。",
    )

    add_para(doc, "(4) 元學習器最佳化 (Optimization of meta-learners)。", bold=True, indent_cm=0)
    add_para(
        doc,
        "當取得所有真實標籤後，本架構計算最終的預測誤差 L_test，以便於"
        "上層最佳化兩個元學習器，亦即資料適配器（DA）與模型適配器（MA）。"
        "元學習器的參數由 φ⁽ᵏ⁻¹⁾ 與 ψ⁽ᵏ⁻¹⁾ 更新為 φᵏ 與 ψᵏ，並用於下一個"
        "增量學習任務。本實作以 Adam 進行更新，MA 學習率 η_φ = 1e-3，"
        "DA 學習率 η_ψ = 1e-3。",
    )

    add_heading(doc, "4.4 訓練流程", level=2)
    add_para(
        doc,
        "完整訓練一次串連三個階段，對應 DoubleAdapt 原論文 Algorithm 1 中"
        "的三個元學習資料集。三階段間共享同一組 (φ, ψ) 參數，前一階段的"
        "最終狀態即為下一階段的初始狀態。任務步長 r 為增量資料與測試資料"
        "之時間跨度，本專題設為 20 個交易日，亦即每個任務之支持集（增量"
        "資料）與查詢集（測試資料）各跨越 20 個交易日。下文依序說明三個"
        "階段。",
    )
    add_para(
        doc,
        "階段 1，離線預訓練（2016-2021，元訓練集）。以 FOMAML 雙層更新"
        "從面板隨機抽取 1000 個樣本 / epoch，內層在支持集（過去 r 個"
        "交易日）以梯度下降得到 θ，外層在查詢集（接續 r 個交易日）以 θ "
        "計算 MSE 損失並以 Adam 更新 (φ, ψ)。每 epoch 結束在 2022-2023 "
        "驗證集上計算 IC，當連續 ζ = 8 個 epoch 無進步即提早停止；"
        "其中 ζ 為「提早停止容忍輪數」參數，意為驗證 IC 連續 ζ 個 "
        "epoch 未進步即終止訓練。提早停止機制的主要目的是防止元學習器"
        "在被部署到實際線上環境之前，先對歷史離線資料過度擬合。結束"
        "訓練後還原至最佳驗證 IC 之權重，即「最佳狀態還原」步驟。本"
        "專題消融實驗顯示此設計為關鍵。",
    )
    add_para(
        doc,
        "階段 2，於驗證集上持續進行增量學習以評估元學習器的調適品質"
        "（2022-2024，重新走過驗證集）。以 r = 20 交易日為步長前推，"
        "共 25 次滾動 FOMAML 步，於驗證集上持續執行與部署時相同的優化器"
        "更新，藉此在進入線上階段前驗證元學習器在最新分布下的調適能力。",
    )
    add_para(
        doc,
        "階段 3，正式線上部署 DoubleAdapt 並於每個線上任務結束時增量"
        "更新元學習器（2024-至今，元測試集）。從階段 2 結束時之權重繼續"
        "以 r = 20 為步長前進，每步做一次 FOMAML 雙層更新並同步進行 "
        "Top-K Dropout 投組模擬，產出部署用之最終模型權重與每日預測"
        "檔案。階段 3 的特色是「邊回測邊微調」── 模型於每個線上任務"
        "結束時吸收新資料訊號，與實際線上部署同步。",
    )

    # ─────────────────────────────────────────────────────────────
    # §5 交易策略
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "5. 交易策略", level=1)

    add_heading(doc, "5.1 投資組成規則", level=2)
    add_para(
        doc,
        "投組規則沿用 qlib 的 Top-K Dropout 策略：每日依模型預測排序取前 "
        "10 名作為目標部位，每日最多替換 1 檔（換手率約 10%）。依 qlib 慣例"
        "保留 5% 現金緩衝（風險投入比 0.95）、每筆買入分配「（現金 + 賣出"
        "可得款）× 0.95 / 買入檔數」的資金。預測分數需高於成本閾值（70 個"
        "基點，約對應實際 0.7% 報酬上限）才會進場，避免雜訊型交易。",
    )
    add_para(
        doc,
        "選擇 top-K = 10 而非 qlib 預設的 50，是因為台股 TW50 宇宙只有 "
        "約 50 檔活躍股，若 top-K = 50 則退化為「持有整個宇宙」失去選股意義。"
        "10/50 = 20% 的選擇性介於 qlib CSI300（50/300 ≈ 17%）與 CSI500"
        "（50/500 = 10%）之間。每日替換 1 檔 / top-K = 10 = 10% 日換手率，"
        "與 qlib 一致。",
    )

    add_heading(doc, "5.2 自動下單系統", level=2)
    add_para(
        doc,
        "每日於課程交易系統收單前約 1 小時執行一次自動下單腳本。流程分為"
        "四個階段："
        "(1) 載入實際部屬模型權重、特徵標準化統計、當期 TW50 活躍宇宙與投組"
        "設定；"
        "(2) 對 50 檔候選代號取昨日為止的歷史日線資料，計算 Alpha360 特徵"
        "(50, 60, 6)，套用與訓練時相同的標準化後由模型推論得到當日預測排序；"
        "(3) 透過課程交易系統 API 取得當前持股，計算需賣出（被踢出前 10 名）"
        "與需買入（新進前 10 名）的部位差異，限價依昨日收盤價套用台股跳動單位"
        "規則向上 / 向下取整化；"
        "(4) 透過 API 逐筆提交委託，將每筆委託結果（成功 / 失敗 / 退回）紀錄至"
        "當日交易紀錄檔。",
    )
    add_para(
        doc,
        "為符合台股每日下單規則（須於 13:00 前下單；15:30~16:00 系統結算"
        "暫停），系統設計為單次推論，不重試失敗的委託（失敗視為次日再嘗試"
        "的訊號），不採槓桿、不做當沖、不做零股。委託在當日 [最低, 最高] "
        "價格區間內成交，否則自動取消。",
    )

    # ─────────────────────────────────────────────────────────────
    # §6 實驗設計
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "6. 實驗設計", level=1)

    add_heading(doc, "6.1 評估指標", level=2)
    add_para(
        doc,
        "本實驗採用 qlib 慣用的四類指標。預測品質類：IC（每日跨截面 Pearson "
        "相關係數的平均值）、ICIR（IC 的平均除以標準差）、RankIC（Spearman 等"
        "級相關平均）、RankICIR。投組績效類：累積報酬、年化 夏普指數 (Sharp)、"
        "最大回撤（max drawdown, MDD）、勝率、獲利因子。對照基準採用"
        "0050.TW（元大台灣 50 ETF，自 2003 年起追蹤 FTSE TWSE 50 指數）"
        "作為投資人最容易取得的被動基準。",
    )

    add_heading(doc, "6.2 雙回測引擎", level=2)
    add_para(
        doc,
        "本專題使用兩套回測引擎並列分析，原因是課程提供的回測完全忽略"
        "交易手續費，無法反映台股實際交易成本（約 0.58% 雙邊往返）對"
        "策略損益 的真實影響。為了能與 DoubleAdapt 原論文中的數據比較，"
        "並確保所得指標的真實性，本專題另行實作了一套對齊原論文與 qlib"
        "慣例的 TWSE 成本回測引擎：",
    )
    add_para(
        doc,
        "qlib 風格引擎：依台股實際成本（手續費 0.1425% × 雙邊 + 賣出稅 0.3%，"
        "每筆至少 NT$20，全程約 0.58% 雙邊往返成本）以當日收盤價成交，"
        "always-fill。與部署現實相比有 1 日先見偏誤（用當日收盤決定當日"
        "下單），但提供完整的日別權益曲線供 夏普指數、最大回撤等風險指標計算。"
        "此設計與 DoubleAdapt 原論文及 qlib 公開的多項基準保持一致。",
    )
    add_para(
        doc,
        "課程框架引擎：透過課程提供之回測模組重新執行預測。零交易成本、"
        "有 1 日因果延遲（當日交易使用前日預測），與線上部署相同因果。"
        "僅輸出最終損益，不暴露日別權益。",
    )
    add_para(
        doc,
        "兩引擎結果並列可揭示「成本」與「因果延遲」的個別影響 ── 例如"
        "實際部屬模型在 qlib 風格引擎得 +56.9%，課程框架引擎得 +49.5%，"
        "差異主要來自 TWSE 成本。"
        "課程引擎的數字反映「無成本理想情境」，qlib 風格引擎的數字才是"
        "可與原論文及實際交易直接比較的真實損益。",
    )

    add_heading(doc, "6.3 實驗設定", level=2)
    add_para(
        doc,
        "本專題共完成多次受控實驗，依目的可分為四類：(1) 基線校準階段：從 "
        "qlib 預設組態出發，逐步調整 epochs、ζ、η_ψ、每日替換檔數與驗證視窗，"
        "並切換至 TEJ 還原資料與官方 TW50/TM100 月度宇宙快照；(2) 部署模型"
        "確認：將標籤視窗改為 1 日、宇宙固定為 TW50、加上線上學習流程，得到"
        "實際部屬模型；(3) 受控消融：以實際部屬模型為基準，逐一更改宇宙、"
        "η_ψ、標籤視窗、截面排序標準化、最佳狀態還原、ζ、FOMAML 任務步長 r、"
        "特徵標準化模式等設計元件；(4) 增量學習基線：完全去除 DoubleAdapt 兩個"
        "適配器，量化兩個適配器的具體貢獻。所有實驗皆使用相同的亂數種子以"
        "確保可重現。",
    )

    # ─────────────────────────────────────────────────────────────
    # §7 結果分析
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "7. 結果分析", level=1)

    add_heading(doc, "7.1 主要結果", level=2)
    add_para(
        doc,
        "實際部屬模型（TW50 宇宙、1 日標籤、線上學習流程、截面排序標準化 "
        "(Cross-Sectional Rank Normalization) 包裝、"
        "η_ψ = 1e-3、r = 20、qlib-exact Alpha360）於 2024-01-02 至 2026-04-30 共 "
        "560 個交易日的回測表現如下：累積報酬 +56.9%（NT$100M → NT$156.7M）、"
        "夏普指數 (Sharp) 1.242、最大回撤 −14.07%、勝率 59.4%、"
        "獲利因子 (profit factor) 2.10，共 1,128 筆成交。其中最大回撤 −14.07% "
        "為全部實驗中最低，夏普指數 1.24 僅次於兩組 20 日標籤之高累積報酬"
        "實驗設定。為求公平比較，本專題假設所有實驗皆採用相同的線上學習"
        "回測流程，差異僅在訓練超參數與資料設定。",
    )
    add_image(doc, FIG / "equity_curves.png", width_cm=15.5)
    add_caption(
        doc,
        "圖 2：策略權益曲線與 0050 ETF 基準對比"
        "（實際部屬模型、高累積報酬實驗、qlib-exact 特徵、"
        "增量學習基線、0050 ETF；起始 = 100）",
    )
    add_para(
        doc,
        "圖 2 中「高累積報酬」實驗設定的累積報酬達 +205%，遠勝實際部屬"
        "模型，但本專題仍未選擇此設定作為部署模型，主因為三點：(1) 該"
        "設定的最大回撤達 −36.4%（實際部屬模型為 −14.07%），下行風險過大，"
        "於課程僅約 1 個月部署期間若遇市場急跌可能單日抹去長期累積報酬；"
        "(2) 該設定採用 20 日預測標籤等多項與 DoubleAdapt 原論文不同的設定，"
        "雖然絕對報酬亮眼但難以與原論文及後續消融做量化比較；(3) 課程要求"
        "部署期間內完成至少 100 筆成交，該設定的低換手、高單筆部位特性"
        "在短期內難以累積足夠筆數，風險與時程皆不利於課堂截止期限。"
        "實際部屬模型的高勝率 (59.4%)、低回撤 (−14.07%)、穩定換手"
        "(1,128 筆 / 560 日) 三項特性更適合可控的短期部署。",
    )
    add_para(
        doc,
        "在訓練過程方面，實際部屬模型預訓練最佳 val IC 為 0.0507（epoch 1），"
        "之後 8 個 epoch 內未再超越，提早停止於 epoch 9 觸發並還原至最佳權重。"
        "進入階段 2 驗證集上的持續增量學習時，階段 2 損失從 1.00 起穩定維持，無需修復 ── "
        "與「跳過最佳狀態還原」對照組的階段 2 損失起點 1.43 形成對比，驗證了"
        "最佳狀態還原設計的必要性。線上學習階段共 28 步，整體 IC = 0.046、"
        "RankIC = 0.067、ICIR = 0.206、RankICIR = 0.326。",
    )
    add_image(doc, FIG / "ic_trajectory.png", width_cm=15.5)
    add_caption(doc, "圖 3：實際部屬模型線上學習階段每 20 日步的 IC 軌跡（共 28 步）")
    add_para(
        doc,
        "由圖 3 可見，28 步中有 7 步為負 IC（占 25%），但移動平均始終維持"
        "在正值區間。整體 IC 0.046 在 1 日標籤下屬正常水準（1 日 IC 結構上"
        "比 20 日 IC 小 3-5 倍）。值得注意的是 2024 Q1（步 1）的負 IC（−0.061）"
        "對應市場急轉折，與模型的階段 2 適配能力有關。",
    )
    add_para(
        doc,
        "相對於 0050 ETF 基準，實際部屬模型累積報酬 +56.9% 落後 0050 的"
        " +184.6%（0050 於該視窗有 TSMC 帶動的史詩級多頭行情）。從超額"
        "報酬角度本模型為 −28.8% 年化、IR −1.69 ── 此並非策略不佳，而是"
        "1 日標籤的保守倉位決策本質上難在指數狂飆時跑贏 buy-and-hold。"
        "但本模型的 −14% 最大回撤明顯優於 0050 的同期回撤，於跌勢中保護力"
        "更佳。",
    )
    add_para(
        doc,
        "在交易行為層面，模型共 1,128 筆成交，勝率 59.4%（全部實驗最高之一），"
        "獲利因子 2.10。平均單筆獲利 +1.85%、平均單筆虧損 −1.42%、"
        "獲利交易與虧損交易的比例約 1.46:1（同樣為全部實驗最高之一）。"
        "TW50 個股的集中度造成資金主要流向 TSMC（2330）、聯發科（2454）、"
        "鴻海（2317）等大型權值股，與台股 ETF 集中度結構一致。",
    )

    add_heading(doc, "7.2 消融研究", level=2)
    add_para(
        doc,
        "為釐清 DoubleAdapt + Alpha360 各設計成分的台股場景貢獻，本節將 "
        "10 次消融以「設計類別」分組呈現。基準為實際部屬模型，其他列改變"
        "單一參數。",
    )
    add_image(doc, FIG / "ablation_bars.png", width_cm=15.5)
    add_caption(doc, "圖 4：18 次實驗的累積報酬、夏普指數、最大回撤排序")

    add_heading(doc, "7.2.1 特徵與標籤標準化", level=3)
    add_para(
        doc,
        "另一組「移除截面排序標準化」的對照設定 ── 讓標籤適配器 H 直接"
        "操作於原始未標準化標籤 (y_raw，即原始次日報酬)，其他不變。結果："
        "預訓練 val IC 幾乎相同（0.050 vs 0.051，Adam 的逐參數變異數標準化"
        "吸收了約 360 倍的損失尺度收縮），但線上增量學習階段崩潰 ── "
        "夏普指數 (Sharp) 從 1.24 減半至 0.43、最大回撤從 −14% 擴大至 −40%、"
        "成交數減半至 566（原始標籤預測在 ±0.01-0.05 區間幾乎全卡在"
        "成本門檻 (進場最低預測門檻, 即進場最低預測門檻，本專題設為 70 個基點 "
        "= 0.7%) 之下）。其根本原因為 FOMAML 內層步崩潰為無更新狀態"
        "（一般 SGD × 內層學習率 1e-3 在 360 倍小的損失上產生近零更新）。"
        "結論：原論文預設（移除 qlib 穩定性技巧）需同步重縮內層學習率才合理，"
        "否則嚴格較差。",
    )
    add_para(
        doc,
        "另一組改用 qlib-exact Alpha360 標準化的設定（5 個價格欄位皆除以"
        "當日收盤 close[t]、成交量除以當日成交量 volume[t]，相對於基準的"
        "「每欄位獨立歸一化」模式），其他不變。預訓練 val IC 提升 +20%"
        "（0.0610 vs "
        "0.0507）、線上 IC 提升 +10%、RankICIR 提升 +17%，但交易損益 "
        "損益 反而下降 3.4 %（累積 +53.4% vs +56.7%）、"
        "夏普指數持平（1.23 vs 1.24）、最大回撤略增（−16.5% vs −14.1%）。"
        "勝率 +2.9 %、獲利因子 +0.07，每筆品質確實提升但 top-K=10 的"
        "策略容量已達上限，更佳的特徵無法轉換為更多累積報酬。新訓練的預設"
        "已切換為 qlib-exact 模式。",
    )

    add_heading(doc, "7.2.2 超參數及 early stopping 的參數調整", level=3)
    add_para(
        doc,
        "「跳過最佳狀態還原」的對照組 ── 不執行預訓練結束時的「最佳狀態還原 "
        "(best state restore)」步驟，"
        "階段 2從第 9 個 epoch（驗證 IC = 0.012）的劣勢權重啟動，而非"
        "第 1 個 epoch 的 0.051 強點。結果：25 個階段 2 步全用於從劣勢初始化"
        "中恢復（階段 2 損失從 1.43 緩降至 1.12，仍未達穩定值），階段 3 繼承"
        "未充分恢復的模型。最終夏普指數 (Sharp) 從 1.24 跌至 0.82、"
        "最大回撤從 −14% 倍增至 −32%、累積報酬 −15 %。結論："
        "最佳狀態還原與 early-stop 為一體機制，缺少其一前者無意義。",
    )
    add_para(
        doc,
        "另一組將提早停止容忍輪數 ζ 從 8 提升至 16 的對照設定 ── 其他完全相同。"
        "ζ 在 early stopping 中扮演「驗證指標連續未進步多少個 epoch 才終止」"
        "的角色：ζ 越大，訓練越「有耐心」等待後段可能出現的更佳權重；"
        "ζ 越小，越早終止以節省 GPU 時間。提早停止機制的主要目的是防止"
        "元學習器在被部署到實際線上環境之前，先對歷史離線資料過度擬合 "
        "(overfitting) ── ζ 即為調節「容忍訓練多久」與「過度擬合風險」"
        "之間的關鍵超參數。預訓練多跑 8 個 epoch"
        "（第 10-17 epoch 的 val IC 在 0.012-0.023 區間漂移，從未接近 "
        "epoch 1 的 0.051），最佳狀態還原仍取 epoch 1，下游所有指標"
        "「位元一致」於基準模型（最終資產 156,725,275.95 元到分），僅多耗約 "
        "6 分鐘 GPU。結論：ζ 僅在預訓練後段找到新 val IC 高點時有意義；"
        "對此 seed / 資料組合 ζ ≥ 2 即等價。",
    )
    add_para(
        doc,
        "另一組將 FOMAML 任務步長 r 從 20 縮短至 5 的設定（階段 2變 97 步、"
        "線上學習變 115 步），其他不變。出現「反直覺結果」：訓練竟然更快"
        "（4.14 分鐘 vs 8.67 分鐘，因每任務僅 250 筆樣本而非 1000，FOMAML "
        "內外層計算成本下降速度超過步數倍增）、預訓練 val IC 反而更高"
        "（0.056 vs 0.051，更多元的任務起始日提升泛化），但交易卻顯著惡化 ── "
        "夏普指數減半（0.65 vs 1.24）、最大回撤倍增（−27.7% vs −14.1%）、"
        "累積報酬 −29 %。原因：每步 IC 的方差倍增（±0.28 vs ±0.13），"
        "預測對短期雜訊過敏。結論：r=20 是部署窗口與預測信心區間的"
        "「匹配甜蜜點」，更頻繁的元更新並非「更好」。",
    )
    add_para(
        doc,
        "另一組將資料適配器 (DA) 學習率 η_ψ 從 1e-3 提升至 1e-2 的設定"
        "（原論文預設為 10 × η_φ 的比率），其他不變。預訓練 val IC 幾乎相同"
        "（0.052 vs 0.051，Adam 吸收學習率變化），"
        "但線上學習每步的 ψ 更新放大 10 倍，跨步預測一致性下降，導致"
        "top-K 在雜訊上輪轉。結果：累積報酬從 +56.7% 跌至 +18.9%"
        "（−38 %）、夏普指數減半（0.51 vs 1.24）、最大回撤 −23% vs"
        "−14%、勝率 −4 %。結論：基準設定的 η_ψ = 1e-3 對線上"
        "學習階段為「承擔性設計」── 原論文預設 10 × 比率對純預訓練無害，"
        "但在滾動微調階段有害。",
    )

    add_heading(doc, "7.2.3 投資宇宙大小", level=3)
    add_para(
        doc,
        "「TM100 宇宙」對照組改用 TM100（FTSE 台灣中型 100 指數，約 100 檔中型股，"
        "TW50 之外的下一 100 名）。TM100 在台股場景對應原論文中 CSI500 的角色 "
        "── 同為「較寬、較低流動性的次層級宇宙」。預訓練驗證 IC 更高 "
        "（0.053 vs 0.020），中型股的多樣性讓模型學到更多模式，但線上"
        "增量學習 IC 反而較低（0.035 vs 0.086），top-K 從 100 中選擇引入更多雜訊。"
        "夏普指數 −0.51、最大回撤 −15 %較差，確認 TW50 為更佳交易宇宙。",
    )
    add_para(
        doc,
        "「BOTH 宇宙」對照組改用 TW50 ∪ TM100 約 150 檔/期，1 日標籤下重複此模式："
        "預訓練驗證 IC 略低（0.041 vs 0.051），線上增量學習 IC 減半（0.026 vs 0.046），"
        "累積報酬 −10 %、夏普指數 −0.25。但最大回撤幾乎相同（−14.01% vs −14.07%），"
        "顯示「較緊密的標籤帶來較緊密的回撤」此規律與宇宙大小無關。",
    )
    add_para(
        doc,
        "另一組同時切換 BOTH 宇宙與 η_ψ = 1e-2 的對角設定。意外發現"
        "「互動效應」：η_ψ 對 TW50 與 BOTH 的傷害程度不同 ── 1e-3 → 1e-2 "
        "在 TW50 上扣 −38 %，在 BOTH 上僅扣 −16 %。換言之，"
        "較寬的宇宙對 ψ 漂移有部分緩衝效果。詳細討論見 §7.3。",
    )
    add_para(
        doc,
        "四個主題各自的關鍵指標分別整理於以下四張表（基準皆為實際部屬模型）。"
        "為與原論文比較，所有 run 皆於同一 2024-01 至 2026-04 操作視窗、"
        "同一 TWSE 成本結構下回測。",
        indent_cm=0,
    )

    add_para(doc, "表 1：特徵與標籤標準化", bold=True, indent_cm=0)
    add_table(
        doc,
        headers=["設定", "預訓練 val IC", "線上 IC", "累積報酬", "夏普指數", "最大回撤"],
        rows=[
            ["實際部屬模型 (per-field + 截面排序標準化)", "0.051", "0.046", "+56.9%", "+1.24", "−14.07%"],
            ["移除截面排序標準化", "0.050", "0.022", "+17.8%", "+0.43", "−39.9%"],
            ["qlib-exact Alpha360 標準化", "0.061", "0.051", "+53.4%", "+1.23", "−16.5%"],
        ],
        col_widths_cm=[6.5, 2.0, 1.5, 1.8, 1.4, 1.6],
        style="Table Grid",
        bold_rows=[0],
    )

    add_para(doc, "表 2：超參數及 early stopping 的參數調整", bold=True, indent_cm=0)
    add_table(
        doc,
        headers=["設定", "預訓練 val IC", "線上 IC", "累積報酬", "夏普指數", "最大回撤"],
        rows=[
            ["實際部屬模型", "0.051", "0.046", "+56.9%", "+1.24", "−14.07%"],
            ["跳過最佳狀態還原", "0.051", "0.010", "+42.2%", "+0.82", "−31.7%"],
            ["提早停止 ζ 從 8 → 16", "0.051", "0.046", "+56.9%", "+1.24", "−14.07%"],
            ["FOMAML 任務步長 r 從 20 → 5", "0.056", "0.045", "+26.4%", "+0.63", "−27.7%"],
            ["DA 學習率 η_ψ 從 1e-3 → 1e-2", "0.052", "0.040", "+18.9%", "+0.51", "−23.0%"],
        ],
        col_widths_cm=[6.5, 2.0, 1.5, 1.8, 1.4, 1.6],
        style="Table Grid",
        bold_rows=[0],
    )

    add_para(doc, "表 3：投資宇宙大小（粗體表示該欄最佳值）", bold=True, indent_cm=0)
    add_table(
        doc,
        headers=["設定", "宇宙", "預訓練 val IC", "累積報酬", "夏普指數", "最大回撤"],
        rows=[
            ["實際部屬模型", "TW50", "0.051", "+56.9%", "+1.24", "−14.07%"],
            ["TM100 中型股 (對應原論文 CSI500)", "TM100", "0.053", "+58.7%", "+0.84", "−33.6%"],
            ["TW50 ∪ TM100，約 150 檔/期", "BOTH", "0.041", "+46.6%", "+0.99", "−14.01%"],
            ["BOTH + η_ψ = 1e-2 (2×2 對角)", "BOTH", "0.041", "+30.8%", "+0.72", "−25.8%"],
        ],
        col_widths_cm=[5.6, 1.4, 2.0, 1.8, 1.4, 1.6],
        style="Table Grid",
        # Bold = best per numeric column:
        #   col 2 (val IC): TM100 = 0.053 (row 1)
        #   col 3 (cum return): TM100 = +58.7% (row 1)
        #   col 4 (Sharp): 實際部屬模型 = +1.24 (row 0)
        #   col 5 (MDD): BOTH = −14.01% (closest to 0; row 2)
        bold_cells=[(1, 2), (1, 3), (0, 4), (2, 5)],
    )

    add_heading(doc, "7.2.4 增量學習基線", level=3)
    add_para(
        doc,
        "為量化 DoubleAdapt 兩個適配器（特徵適配器 G 與標籤適配器 H）的"
        "具體貢獻，本專題新訓練了一個「增量學習基線」作為對照組。"
        "此基線完全移除 G 與 H，僅保留 GRU 基底；預訓練改用一般 MSE"
        "（無 FOMAML 雙層），線上增量學習改用每 20 日 10 步一般 SGD 微調。"
        "其他設定（Alpha360 特徵、截面排序標準化 (Cross-Sectional Rank "
        "Normalization)、Top-K Dropout 投組、TWSE 成本、TW50 宇宙）與"
        "實際部屬模型完全相同。",
    )
    add_para(
        doc,
        "結果：增量學習基線累積報酬 +49.96%（vs 實際部屬模型 +56.9%）、"
        "夏普指數 0.92（vs 1.24）、最大回撤 −26.87%（vs −14.07%）。換言之，"
        "DoubleAdapt 兩個適配器（G + H + FOMAML 雙層）共貢獻 +7 %"
        "累積報酬、+0.32 夏普指數，且最大回撤幾乎減半（縮減 12.8 %）── "
        "DoubleAdapt 雙適配器最強的單項貢獻為下行保護。此驗證 §3 相關工作"
        "提出之核心問題（iii）「相較於增量學習基線，DoubleAdapt 的兩個適配器"
        "具體貢獻多少？」",
    )
    add_para(doc, "表 4：增量學習基線對照", bold=True, indent_cm=0)
    add_table(
        doc,
        headers=["模型架構", "預訓練 val IC", "線上 IC", "累積報酬", "夏普指數", "最大回撤"],
        rows=[
            ["DoubleAdapt 完整（G + F + H + FOMAML）", "0.051", "0.046", "+56.9%", "+1.24", "−14.07%"],
            ["增量學習基線（僅 F，無 G/H，無 FOMAML）", "0.059", "0.037", "+50.0%", "+0.92", "−26.87%"],
        ],
        col_widths_cm=[6.8, 2.0, 1.5, 2.0, 1.4, 2.0],
        style="Table Grid",
        bold_rows=[0],
    )

    add_heading(doc, "7.3 反直覺發現", level=2)
    add_para(doc, "本節整理四項「預期與實證矛盾」的關鍵發現，每項皆有 §7.2 中明確" "支援的消融證據。", indent_cm=0)
    add_para(
        doc,
        "發現 1：預訓練 val IC 並非損益的可靠代理。「TM100 宇宙」設定的"
        "預訓練 val IC = 0.053，是「TW50 + 20 日標籤」設定的 2.6 倍；"
        "「FOMAML 任務步長 r = 5」設定為 0.056，為實際部屬模型的 1.10 倍；"
        "「qlib-exact Alpha360」設定為 0.061，為實際部屬模型的 1.20 倍。"
        "三者皆在預訓練階段「看起來更強」，但下游交易夏普指數分別下降 −0.51、"
        "−0.59、持平。原因：top-K 投組策略對「跨股排序一致性」敏感，"
        "而預訓練 val IC 衡量的是「跨日預測能力」── 兩者並非同一回事。"
        "建議：以線上學習階段的 ICIR 與每步 IC 變異作為更可靠的訓練 KPI。",
        indent_cm=0,
    )
    add_para(
        doc,
        "發現 2：宇宙大小與 η_ψ 存在強互動效應。「TW50 + η_ψ = 1e-3」"
        "（實際部屬模型）為 +56.7%；「TW50 + η_ψ = 1e-2」為 +18.9%"
        "（−38 %）；「BOTH + η_ψ = 1e-3」為 +46.4%；「BOTH + "
        "η_ψ = 1e-2」為 +30.8%（−16 %）。換言之，η_ψ 對 TW50 的"
        "傷害遠大於對 BOTH。「η_ψ = 1e-3 是承擔性設計」其實是「在窄宇宙下"
        "是承擔性」── 較寬的宇宙以橫斷面平均緩衝 ψ 漂移。意涵：「一次只改"
        "一個超參數」的標準消融會錯失此類交叉現象；兩個超參數的最佳值"
        "彼此相依。",
        indent_cm=0,
    )
    add_para(
        doc,
        "發現 3：較緊密的標籤帶來較緊密的回撤。「20 日標籤 + TW50」"
        "設定的最大回撤為 −18.1%，「1 日標籤 + TW50」（實際部屬模型）為 "
        "−14.07%，「1 日標籤 + BOTH」為 −14.01%。1 日標籤的兩個設定都"
        "落在 −14% 區間，與宇宙無關。原因：1 日標籤迫使模型不對 20 日"
        "方向性做出大注，倉位調整更具反應性、更小。代價是累積報酬下降"
        "（中期動能信號錯失），但符合「論文忠實 + 部署誠實」的工程權衡："
        "1 日標籤完全沒有線上學習階段的「標籤洩漏」（次日報酬可隔日觀測），"
        "且最大回撤明確改善。",
        indent_cm=0,
    )
    add_para(
        doc,
        "發現 4：更寬的宇宙不會自動幫 top-K 策略。「TW50 → TM100」"
        "（20 日標籤）與「TW50 → BOTH」（1 日標籤）皆觀察到「寬宇宙 → "
        "預訓練 val IC 較高 → 但線上 IC 較低、夏普指數較低」此一致模式。"
        "原因：更寬的候選池讓模型學到更豐富的橫斷面結構（val IC 上升），"
        "但 top-K 從更雜訊的池中選出反而降低品質。意涵：宇宙大小應與"
        "策略的 top-K 配對選擇；TW50 + top-K = 10 為本實作的最佳組合。",
        indent_cm=0,
    )

    add_heading(doc, "7.4 線上部署結果", level=2)
    add_para(doc, "自 2026-05-27 起以實際部屬模型進行每日自動下單。" "截至 2026-06-26（報告繳交日）為止：")
    add_para(doc, "• 累積投資報酬：+40%（初始 NT$100M → NT$140M）", indent_cm=0)
    add_para(doc, "• 課程排行榜排名：第 8 名", indent_cm=0)
    add_para(doc, "• 累積成交筆數：> 100 筆（符合課程系統最低交易筆數要求）", indent_cm=0)
    add_para(doc, "• 部署模型：本專題之實際部屬模型（TW50、1 日標籤、η_ψ = 1e-3）", indent_cm=0)
    add_para(
        doc,
        "由於線上樣本數仍偏小（~21 個交易日），統計顯著性不足以做嚴格的"
        "「回測 vs 線上一致性」分析，但定性上線上行為與實際部屬模型回測"
        "（−14% MDD、夏普指數 1.24、勝率 59%）一致 ── 無單日大跌、"
        "TSMC、聯發科、鴻海為主要持倉。期間 TW50 權值股整體偏多頭，"
        "策略以中低換手率穩定累積部位。",
    )

    # ─────────────────────────────────────────────────────────────
    # §8 討論與限制
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "8. 討論與限制", level=1)
    add_para(
        doc,
        "限制 1：多頭市場視窗。本實驗的操作視窗 2024-01 → 2026-04 為台股"
        "歷史性的 AI/TSMC 大多頭行情（0050 同期 +184.6%），所有 run 的"
        "「正報酬」皆涵蓋此一外部多頭因素。空頭與震盪市場的行為尚未驗證；"
        "若市場進入長期下跌，DoubleAdapt 的元學習能否快速適應反向漂移"
        "需進一步追蹤。",
        indent_cm=0,
    )
    add_para(
        doc,
        "限制 2：線上部署樣本過小。報告繳交時實際部署僅 ~21 個交易日，"
        "不足以對「回測 vs 線上一致性」做統計檢定。雖定性上一致，"
        "但偶然性的單一行情事件（如某日權值股跳空）可能主導目前的 +40%。"
        "建議至少累積 6 個月（~120 個交易日）才能做嚴格驗證。",
        indent_cm=0,
    )
    add_para(
        doc,
        "限制 3：課堂交易平台存在未來資訊套利可能。本專題實作中發現該平台"
        "允許下單時的限價落在當日 [最低, 最高] 區間即視為成交，配合可"
        "從免費資料源取得約 15 分鐘延遲的當日盤中 OHLC，理論上可"
        "以「當日盤中最低價買、最高價賣」之最佳執行進行套利。本專題的"
        "正式部署一律採用前日收盤推算之限價（嚴格因果，無洩漏），並未利用"
        "此漏洞；但需明確指出此漏洞屬於課堂模擬平台本身的設計限制，"
        "若為實際交易環境並不存在此種跨時資訊。本報告陳述此點僅為說明"
        "評估誠實性，而非建議他人利用此漏洞。",
        indent_cm=0,
    )
    add_para(
        doc,
        "限制 4：基線覆蓋仍不完整。增量學習基線量化了 DoubleAdapt 兩個"
        "適配器的具體貢獻，但仍未對比 Rolling Retraining（RR）── 從零重訓的"
        "標準做法。DoubleAdapt 原論文對 CSI300 報告 RR 較 IL 慢 ~10 倍但僅"
        "些微優於 IL；本實作未複現此比較。此外，亦未對比其他元學習方法"
        "（MAML on GRU、Reptile、OnRL），故僅可宣稱「DoubleAdapt 於台股場景"
        "工作良好」而非「DoubleAdapt 為最佳方法」。",
        indent_cm=0,
    )

    add_para(
        doc,
        "未來工作方向：(1) 多隨機種子驗證（縮減 ±隨機性區間）；"
        "(2) 加入 2022 年空頭區段擴展操作視窗；(3) 完整 RR/Reptile/"
        "MAML 基線比較；(4) 加入因子模型（Fama-French 5）解構"
        "DoubleAdapt 的 alpha 來源；(5) 結合更廣的資產類別"
        "（ETF 套利、轉換公司債）擴展策略容量。",
    )

    # ─────────────────────────────────────────────────────────────
    # §9 結論
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "9. 結論", level=1)
    add_para(
        doc,
        "本專題透過 18 次受控訓練實驗，在台股 TW50 宇宙上完整實作並驗證了"
        "DoubleAdapt 元學習框架。實際部屬模型（TW50、1 日標籤、"
        "線上學習流程、截面排序標準化 (Cross-Sectional Rank Normalization) 包裝、η_ψ=1e-3、r=20）"
        "於 2024-01 至 "
        "2026-04 的回測獲得 夏普指數 1.24、最大回撤 −14.07%（全 18 次最低），"
        "並成功部署為每日自動下單系統，截至 2026-06-26 線上累積報酬 +40%、"
        "課程排行榜第 8 名。",
    )
    add_para(
        doc,
        "10 項受控消融揭示了 DoubleAdapt 各設計元件的台股場景貢獻："
        "(a) 截面排序標準化 (Cross-Sectional Rank Normalization) 為「承擔性穩定技巧」，"
        "移除會使夏普指數減半；"
        "(b) best-state restore 與 early-stop 為一體機制，缺一不可；"
        "(c) η_ψ=1e-3 為窄宇宙的承擔性設計，但對寬宇宙傷害較小；"
        "(d) r=20 為部署窗口與預測信心區間的匹配甜蜜點；"
        "(e) 1 日標籤雖犧牲 25 pp 累積報酬但換得最低回撤與部署誠實性。"
        "與無 DA 之增量學習基線相比，DoubleAdapt 的雙適配器貢獻 "
        "+0.32 夏普指數 與 12.8 pp 最大回撤改善 ── 主要價值為下行保護。",
    )
    add_para(
        doc,
        "本實作驗證了 DoubleAdapt 從中國 A 股 CSI300/CSI500 跨市場至"
        "台灣 TW50 的有效性；亦記錄了在較小宇宙下的若干新發現 ── 特別是"
        "宇宙大小 × η_ψ 的互動效應（標準「單變量消融」會錯失此類交叉"
        "現象），以及預訓練 val IC 與交易損益 的脫節（top-K 容量限制"
        "下，更佳特徵未必轉換為更高累積報酬）。",
    )

    # ─────────────────────────────────────────────────────────────
    # 參考文獻
    # ─────────────────────────────────────────────────────────────
    add_heading(doc, "參考文獻", level=1)
    refs = [
        '[1] L. Zhao, S. Kong, and Y. Shen, "DoubleAdapt: A Meta-learning '
        'Approach to Incremental Learning for Stock Trend Forecasting," in '
        "Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery "
        "and Data Mining (KDD '23), Long Beach, CA, USA, Aug. 6-10, 2023, "
        "pp. 3492-3505. https://doi.org/10.1145/3580305.3599315",
        '[2] C. Finn, P. Abbeel, and S. Levine, "Model-Agnostic Meta-Learning '
        'for Fast Adaptation of Deep Networks," in Proceedings of the 34th '
        "International Conference on Machine Learning (ICML '17), Sydney, "
        "Australia, Aug. 6-11, 2017, pp. 1126-1135.",
        '[3] X. Yang, W. Liu, D. Zhou, J. Bian, and T.-Y. Liu, "Qlib: An '
        'AI-oriented Quantitative Investment Platform," arXiv preprint '
        "arXiv:2009.11189, 2020. https://github.com/microsoft/qlib",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.3
        r = p.add_run(ref)
        set_cjk(r, size=10)

    # ─────────────────────────────────────────────────────────────
    # 附錄 A 超參數表
    # ─────────────────────────────────────────────────────────────
    add_page_break(doc)
    add_heading(doc, "附錄 A", level=1)
    add_para(doc, "下表為實際部屬模型的完整超參數，按類別分組：")

    add_table(
        doc,
        headers=["類別", "參數", "值", "說明"],
        rows=[
            ["資料", "資料起始", "2016-01-01", "資料池起始日"],
            ["資料", "資料結束", "2026-05-23 (動態)", "資料池上界"],
            ["資料", "預訓練視窗", "2016-01-01 ~ 2021-12-31", "Meta train set (6 年)"],
            ["資料", "驗證視窗", "2022-01-01 ~ 2023-12-31", "Meta valid set (2 年)"],
            ["資料", "線上學習視窗", "2024-01-01 ~ 面板最大日", "Meta test set (約 2.5 年)"],
            ["資料", "投資宇宙", "TW50", "TEJ 官方月度成分快照"],
            ["特徵", "歷史視窗長度", "60", "每筆樣本回望日數"],
            ["特徵", "欄位數", "6", "收盤、開盤、最高、最低、VWAP、成交量"],
            ["特徵", "特徵歸一化模式", "per-field", "Alpha360 標準化模式"],
            ["特徵", "預測視窗長度", "1", "次日收盤報酬"],
            ["特徵", "穩健 Z-Score 標準化", "中位數 + MAD", "在預訓練面板上估計"],
            ["特徵", "截面排序標準化", "每日排序 → 標準常態", "每任務內標籤包裝"],
            ["模型", "GRU 輸入維度", "6", ""],
            ["模型", "GRU 隱藏維度", "64", ""],
            ["模型", "GRU 層數", "2", ""],
            ["模型", "特徵適配器原型數 N", "8", ""],
            ["模型", "特徵適配器溫度 τ", "10.0", "softmax 溫度"],
            ["模型", "標籤適配器頭數", "8", ""],
            ["模型", "標籤適配器隱藏維度", "32", ""],
            ["模型", "DA 參數總數", "12,176", "G + H 合計"],
            ["模型", "MA 參數總數", "38,849", "GRU 基底"],
            ["訓練", "FOMAML 任務步長 r", "20", "交易日"],
            ["訓練", "FOMAML 內層學習率 η_θ", "1e-3", "θ 單步更新"],
            ["訓練", "FOMAML 正則項權重 α", "0.5", "L_reg 權重"],
            ["訓練", "FOMAML MA 學習率 η_φ", "1e-3", "Adam"],
            ["訓練", "FOMAML DA 學習率 η_ψ", "1e-3", "Adam (覆蓋原論文 1e-2 預設)"],
            ["訓練", "預訓練 epochs 上限", "30", "實際提早停止於 epoch 9"],
            ["訓練", "每 epoch 樣本數", "1000", "= r × 投資宇宙大小"],
            ["訓練", "提早停止容忍輪數 ζ", "8", "驗證 IC 連續未進步輪數上限"],
            ["訓練", "亂數種子", "0", "torch manual seed"],
            ["回測", "起始資本", "1e8", "NT$100M"],
            ["回測", "top-K", "10", "每日持有檔數"],
            ["回測", "每日替換檔數", "1", "日換手率約 10%"],
            ["回測", "進場最低預測門檻", "70", "基點 (約 0.7%)"],
            ["回測", "成本模式", "twse", "0.1425% × 2 + 0.3% 稅"],
            ["回測", "成交價", "close", "今日收盤"],
            ["回測", "風險投入比", "0.95", "保留 5% 現金緩衝"],
            ["回測", "漲跌停閾值", "0.095", "封漲跌停"],
        ],
        col_widths_cm=[1.4, 5.6, 3.5, 5.4],
        style="Table Grid",
    )

    add_heading(doc, "附錄 B", level=1)
    add_para(
        doc,
        "為便於對照本實作與原論文之差異，以下重製 DoubleAdapt 原論文中"
        "之兩個關鍵演算法（取自原論文圖表）。圖 A1 為離線預訓練流程"
        "（對應本實作階段 1），圖 A2 為線上學習流程（對應本實作"
        "階段 2 與階段 3）。本實作在 (φ, ψ) 更新、early-stop、最佳狀態還原、"
        "與截面排序標準化 (Cross-Sectional Rank Normalization) 包裝等細節上對原論文流程"
        "有微幅調整，詳見 §4 與 §7.2。",
    )
    _alg1 = FIG / "algorithm1.png"
    if not _alg1.exists():
        _alg1 = PROJECT_ROOT / "docs" / "algorithm1.png"
    _alg2 = FIG / "algorithm2.png"
    if not _alg2.exists():
        _alg2 = PROJECT_ROOT / "docs" / "algorithm2.png"
    if _alg1.exists():
        add_image(doc, _alg1, width_cm=15.0)
        add_caption(doc, "圖 A1：DoubleAdapt 原論文 Algorithm 1 ── 離線預訓練流程")
    else:
        add_para(doc, "[待補:algorithm1.png 圖檔]", indent_cm=0)
    if _alg2.exists():
        add_image(doc, _alg2, width_cm=15.0)
        add_caption(doc, "圖 A2：DoubleAdapt 原論文 Algorithm 2 ── 線上學習流程")
    else:
        add_para(doc, "[待補:algorithm2.png 圖檔]", indent_cm=0)

    doc.save(OUT_PATH)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    build_report()
