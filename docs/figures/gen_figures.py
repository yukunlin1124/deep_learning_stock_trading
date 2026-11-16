"""Generate report figures from existing run artifacts.

Produces 3 PNG + PDF figures in docs/figures/ ready to paste into Word:
  1. equity_curves.png  -- run9 + run3 + run17 + run18 (if present) vs 0050
  2. ic_trajectory.png  -- run9's live-IL per-step IC over time
  3. ablation_bars.png  -- all runs sorted by Sharpe, color-coded by family

Mandarin axis labels supported on Windows via Microsoft JhengHei.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_BASE = PROJECT_ROOT / "models"

OP_WINDOW_START = "2024-01-02"
OP_WINDOW_END = "2026-04-30"

# Match the docx style: 標楷體 (DFKai-SB) for CJK + Latin both.
# matplotlib's findfont() can't see DFKai-SB by default on Windows (it's a
# .ttf but lives in C:/Windows/Fonts only); addfont() registers it explicitly.
_KAI_PATHS = [
    Path("C:/Windows/Fonts/kaiu.ttf"),  # standard Windows 標楷體
    Path("C:/Windows/Fonts/DFKAI-SB.ttf"),
]
for _p in _KAI_PATHS:
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))
        matplotlib.rcParams["font.family"] = "DFKai-SB"
        print(f"using CJK font: DFKai-SB ({_p.name})")
        break
else:
    # Fallback to whatever CJK font is available.
    for font_name in ["Microsoft JhengHei", "Noto Serif CJK TC"]:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = font_name
            print(f"using fallback CJK font: {font_name}")
            break
        except Exception:
            continue
matplotlib.rcParams["axes.unicode_minus"] = False

PALETTE = {
    "run9": "#1f77b4",
    "run3": "#ff7f0e",
    "run17": "#2ca02c",
    "run18": "#d62728",
    "0050": "#7f7f7f",
}


def load_equity(run: str) -> pd.DataFrame | None:
    p = OUTPUT_BASE / run / "equity.csv"
    if not p.exists():
        print(f"  skip {run}: no equity.csv")
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[(df["date"] >= OP_WINDOW_START) & (df["date"] <= OP_WINDOW_END)]
    if df.empty:
        return None
    df["norm"] = df["equity"] / df["equity"].iloc[0] * 100.0
    return df.reset_index(drop=True)


def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "0050_benchmark.csv", parse_dates=["date"])
    df = df[(df["date"] >= OP_WINDOW_START) & (df["date"] <= OP_WINDOW_END)]
    df["norm"] = df["close"] / df["close"].iloc[0] * 100.0
    return df.reset_index(drop=True)


def load_summary(run: str) -> dict | None:
    p = OUTPUT_BASE / run / "report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r.get("backtest", r)


def figure_1_equity_curves():
    print("[figure 1] equity_curves")
    fig, ax = plt.subplots(figsize=(9, 5))

    bm = load_benchmark()
    ax.plot(bm["date"], bm["norm"], color=PALETTE["0050"], linewidth=1.5, linestyle="--", label="0050 ETF (基準)")

    # Fresh cum-return values from the 2026-05-28 cross-run replay
    # (model_env.md) — overrides report.json which can be stale.
    FRESH_CUM = {"run9": 56.9, "run3": 205.2, "run17": 53.4, "run18": 50.0}
    for run in ["run9", "run3", "run17", "run18"]:
        eq = load_equity(run)
        if eq is None:
            continue
        cum_pct = FRESH_CUM.get(run, (eq["norm"].iloc[-1] - 100.0))
        label = f"{run}  (累積 {cum_pct:+.1f}%)"
        ax.plot(eq["date"], eq["norm"], color=PALETTE.get(run, "#333"), linewidth=1.8, label=label)

    ax.set_xlabel("日期")
    ax.set_ylabel("資產 (起始 = 100)")
    ax.set_title("策略權益曲線 vs 0050 ETF 基準  (2024-01 → 2026-04)")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)
    ax.axhline(100, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curves.png", dpi=200)
    fig.savefig(OUT_DIR / "equity_curves.pdf")
    plt.close(fig)


def figure_2_ic_trajectory():
    print("[figure 2] ic_trajectory (run9)")
    p = OUTPUT_BASE / "run9" / "il_log.csv"
    if not p.exists():
        print("  skip: run9/il_log.csv missing")
        return
    df = pd.read_csv(p)
    test_start_col = "test_start" if "test_start" in df.columns else "query_start"
    df["test_start"] = pd.to_datetime(df[test_start_col])
    df = df[df["test_start"] >= OP_WINDOW_START].reset_index(drop=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["ic"]]
    ax.bar(df["test_start"], df["ic"], width=15, color=colors, alpha=0.85, label="每步 IC")
    rolling = df["ic"].rolling(window=3, min_periods=1, center=True).mean()
    ax.plot(df["test_start"], rolling, color="#1f77b4", linewidth=2, label="3-step 移動平均")

    mean_ic = df["ic"].mean()
    ax.axhline(mean_ic, color="black", linewidth=1, linestyle=":", label=f"整體均值 IC = {mean_ic:+.4f}")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.set_xlabel("線上訓練測試窗口開始日")
    ax.set_ylabel("跨截面 IC (Pearson)")
    ax.set_title(f"run9 線上學習：每 20 日步的 IC 軌跡  ({len(df)} 步)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ic_trajectory.png", dpi=200)
    fig.savefig(OUT_DIR / "ic_trajectory.pdf")
    plt.close(fig)


def figure_3_ablation_bars():
    """Uses the FRESH cross-run replay numbers (matching docs/model_env.md)
    rather than each run's stale report.json — the latter is overwritten only
    when the run is re-trained, while the cross-run replay uses unified bars.
    run18 added from its own report.json since it was just trained."""
    print("[figure 3] ablation_bars")

    # Cross-run replay values from docs/model_env.md (2026-05-28 refresh) +
    # run17/run18 fresh report.json. cum_ret_pct, sharpe, max_dd_pct
    FRESH = {
        "run1": (88.5, 1.066, -38.6),
        "run2": (147.7, 1.373, -45.5),
        "run3": (205.2, 1.661, -36.4),
        "run4": (131.7, 1.427, -40.7),
        "run5": (156.8, 1.417, -43.6),
        "run6": (94.7, 1.520, -19.3),
        "run7": (79.2, 1.367, -18.1),
        "run8": (58.7, 0.842, -33.6),
        "run9": (56.9, 1.242, -14.07),
        "run10": (46.6, 0.986, -14.01),
        "run11": (17.8, 0.425, -39.91),
        "run12": (42.2, 0.822, -31.66),
        "run13": (56.9, 1.242, -14.07),
        "run14": (26.4, 0.627, -27.71),
        "run15": (18.9, 0.510, -23.02),
        "run16": (30.8, 0.720, -25.80),
        "run17": (53.4, 1.227, -16.5),
        "run18": (50.0, 0.924, -26.87),
    }
    rows = [{"run": r, "cum_return_pct": v[0], "sharpe": v[1], "max_dd_pct": v[2]} for r, v in FRESH.items()]
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, 0.32 * len(df))), sharey=True)
    y = np.arange(len(df))
    color = ["#1f77b4" if r == "run9" else "#9ac9e3" for r in df["run"]]

    axes[0].barh(y, df["cum_return_pct"], color=color)
    axes[0].set_xlabel("累積報酬 (%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["run"])
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y, df["sharpe"], color=color)
    axes[1].set_xlabel("Sharpe Ratio")
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].grid(alpha=0.3, axis="x")

    axes[2].barh(y, df["max_dd_pct"], color=color)
    axes[2].set_xlabel("最大回撤 (%)  ← 越接近 0 越好")
    axes[2].axvline(0, color="black", linewidth=0.5)
    axes[2].grid(alpha=0.3, axis="x")

    fig.suptitle("17 次實驗 — 三項指標排序 (run9 為基準藍色)", y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ablation_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"output dir: {OUT_DIR}")
    figure_1_equity_curves()
    figure_2_ic_trajectory()
    figure_3_ablation_bars()
    print("done.")


if __name__ == "__main__":
    main()
