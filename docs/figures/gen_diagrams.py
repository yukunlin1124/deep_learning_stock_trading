"""Generate architecture + training-timeline diagrams as PNG via matplotlib.

Replaces Mermaid render path (no Node/Chrome dependency required).
Outputs: docs/figures/architecture.png, training_timeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent
# Match the docx style: 標楷體 (DFKai-SB on Windows).
_KAI_PATHS = [
    Path("C:/Windows/Fonts/kaiu.ttf"),
    Path("C:/Windows/Fonts/DFKAI-SB.ttf"),
]
for _p in _KAI_PATHS:
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))
        matplotlib.rcParams["font.family"] = "DFKai-SB"
        print(f"using CJK font: DFKai-SB ({_p.name})")
        break
else:
    for font_name in ["Microsoft JhengHei", "Noto Serif CJK TC"]:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = font_name
            break
        except Exception:
            continue
matplotlib.rcParams["axes.unicode_minus"] = False


def _box(ax, xy, w, h, text, color, edge="#333", fontsize=18, fontweight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15", facecolor=color, edgecolor=edge, linewidth=2.0
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, wrap=True)


def _arrow(ax, src, dst, label=None, color="#333", style="-|>"):
    arr = FancyArrowPatch(
        src, dst, arrowstyle=style, mutation_scale=22, linewidth=2.0, color=color, shrinkA=4, shrinkB=4
    )
    ax.add_patch(arr)
    if label:
        mx, my = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
        ax.text(mx + 0.05, my + 0.25, label, fontsize=15, color=color, ha="left", va="bottom", style="italic")


def architecture_diagram():
    # Enlarged 2× from previous version for in-report readability.
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    DA = "#fff4e6"
    MA = "#e6f4ff"
    INPUT = "#f0f0f0"
    OUT = "#e6ffe6"

    _box(ax, (0.3, 3.8), 2.4, 1.4, "輸入 X\n(B, 60, 6)\n6 欄位 × 60 日", INPUT, fontsize=17)

    _box(
        ax,
        (3.4, 6.2),
        4.2,
        2.4,
        "G — 特徵適配器\n每時間步餘弦 (cosine) 相似度\n對 N=8 學習得到的原型\nsoftmax(sims/τ) → 多頭殘差變換",
        DA,
        edge="#d97000",
        fontsize=16,
        fontweight="bold",
    )

    _box(ax, (3.4, 3.8), 4.2, 1.4, "x̃ (調整後特徵)", INPUT, fontsize=17)

    _box(
        ax,
        (8.0, 3.4),
        3.0,
        2.2,
        "F — GRU 基底\nGRU(in=6, hid=64, L=2)\n→ Linear(64,1)\n→ pred_raw (scalar)",
        MA,
        edge="#0066cc",
        fontsize=16,
        fontweight="bold",
    )

    _box(
        ax,
        (3.4, 0.4),
        4.2,
        2.4,
        "H — 標籤適配器\n獨立餘弦 (cosine) gating\n每頭仿射 h_i(y) = γ_i·y + β_i\n正向 / 反向皆可用",
        DA,
        edge="#d97000",
        fontsize=16,
        fontweight="bold",
    )

    _box(ax, (11.4, 3.8), 2.4, 1.4, "ŷ (最終預測)\n回到真實尺度", OUT, edge="#00aa00", fontsize=17, fontweight="bold")

    _arrow(ax, (2.7, 4.5), (3.4, 5.5))
    _arrow(ax, (5.5, 6.2), (5.5, 5.2), label="x̃ ")
    _arrow(ax, (7.6, 4.5), (8.0, 4.5))
    _arrow(ax, (5.5, 3.8), (5.5, 2.8), label="    x̃ (gating)")
    _arrow(ax, (11.0, 4.5), (11.4, 4.5))
    _arrow(ax, (9.5, 3.4), (7.6, 1.6), label="    pred_raw")
    _arrow(ax, (7.6, 1.6), (11.7, 3.8), color="#00aa00", label="    H 反向")

    ax.text(
        5.5,
        8.7,
        "DA(ψ) = G + H — 資料適配器（共 12,176 個參數）",
        ha="center",
        fontsize=18,
        color="#d97000",
        fontweight="bold",
    )
    ax.text(
        9.5,
        5.85,
        "MA(φ) — 模型適配器\n（共 38,849 個參數）",
        ha="center",
        fontsize=17,
        color="#0066cc",
        fontweight="bold",
    )

    ax.set_title(
        "DoubleAdapt 架構：特徵適配器 (G) + GRU 基底 (F) + 標籤適配器 (H)", fontsize=22, pad=20, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[architecture] saved")


def training_timeline_diagram():
    fig, ax = plt.subplots(figsize=(28, 14))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    INIT = "#fff4e6"
    S1 = "#e6f4ff"
    S2 = "#e6f9ff"
    S3 = "#e6ffe6"
    DEPLOY = "#ffe6e6"

    _box(ax, (0.1, 2.6), 2.2, 1.8, "模型初始化\n(φ, ψ) 隨機\n~51k 參數", INIT, edge="#d97000", fontsize=22)

    _box(
        ax,
        (2.6, 2.0),
        3.2,
        3.0,
        "階段 1\n離線預訓練\n2016-01 → 2021-12\n驗證: 2022-2023\nFOMAML 雙層更新\n1000 任務 / epoch\nearly-stop on val IC",
        S1,
        edge="#0066cc",
        fontsize=20,
    )

    _box(
        ax,
        (6.1, 2.0),
        3.2,
        3.0,
        "階段 2\n暖機線上學習\n2022-01 → 2024-01\nr=20 滾動 FOMAML\n共 25 次雙層更新\n→ 產生暖機後模型",
        S2,
        edge="#0099cc",
        fontsize=20,
    )

    _box(
        ax,
        (9.6, 2.0),
        3.2,
        3.0,
        "階段 3\n線上學習 + 回測\n2024-01 → 至今\n28 步 FOMAML\n+ 投組模擬\nTWSE 0.58% 成本",
        S3,
        edge="#00aa00",
        fontsize=20,
    )

    _box(ax, (13.1, 2.6), 1.8, 1.8, "每日推論\n+ 自動下單", DEPLOY, edge="#cc0000", fontsize=20)

    _arrow(ax, (2.3, 3.5), (2.6, 3.5))
    _arrow(ax, (5.8, 3.5), (6.1, 3.5), label="φ, ψ best IC")
    _arrow(ax, (9.3, 3.5), (9.6, 3.5), label="φ, ψ 暖機後")
    _arrow(ax, (12.8, 3.5), (13.1, 3.5), label="部署模型")

    ax.text(4.2, 1.3, "meta_train_set", ha="center", fontsize=19, style="italic", color="#555")
    ax.text(7.7, 1.3, "meta_valid_set\n(重新走過)", ha="center", fontsize=19, style="italic", color="#555")
    ax.text(11.2, 1.3, "meta_test_set\n(同步回測)", ha="center", fontsize=19, style="italic", color="#555")

    ax.text(7.5, 6.4, "DoubleAdapt 三階段訓練時程", ha="center", fontsize=30, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[training_timeline] saved")


def main():
    architecture_diagram()
    training_timeline_diagram()
    print("done.")


if __name__ == "__main__":
    main()
