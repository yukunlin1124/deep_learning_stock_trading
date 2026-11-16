"""Portfolio strategy: qlib's TopkDropoutStrategy selection + buy-slot sizing.

One implementation shared by the replay backtest, the live-IL backtest
(evaluate/backtest_qlib.py) and live order planning (scripts/deploy_today.py),
so a strategy change is made once. Each caller still ranks its own target
list (they rank from different containers, and keeping that ranking
unchanged keeps results bit-identical) and does its own pricing/fills.

scripts/run_course_backtest.py is deliberately NOT a user: the course
framework's callbacks implement plain top-K rebalancing (no n_drop).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def topk_dropout(
    target: list[str],
    held: Iterable[str],
    scores: Mapping[str, float],
    top_k: int,
    n_drop: int | None,
) -> tuple[list[str], list[str]]:
    """Today's (sells, buys) given the ranked target list and current holdings.

    target: today's top-k codes, best first (already thresholded by the caller).
    scores: today's prediction per code; held codes with no score rank lowest.
    Sells are held codes outside the target, lowest score first, capped at n_drop.
    Buys follow qlib: len(sells) + (top_k - len(held)), so an empty or partial
    book fills its open slots and a full book replaces n_drop names.

    Known issue: held codes with no score all tie at -inf, and their order comes
    from set iteration, so which of them are sold depends on PYTHONHASHSEED.
    """
    target_set = set(target)
    held_set = set(held)
    sells = sorted(held_set - target_set, key=lambda c: scores.get(c, -np.inf))
    buys = [c for c in target if c not in held_set]
    if n_drop is not None:
        sells = sells[:n_drop]
    buy_budget = len(sells) + (top_k - len(held_set))
    buys = buys[: max(buy_budget, 0)]
    return sells, buys


def buy_slot(cash: float, risk_degree: float, n_buys: int) -> float:
    """qlib's value per buy: cash * risk_degree / len(buys) (0 if no buys)."""
    return (cash * risk_degree / n_buys) if n_buys > 0 else 0.0
