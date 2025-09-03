from typing import List, Dict, Iterable
import numpy as np
import pandas as pd

def nearest_indices_every_other(sorted_values, center_value: float, k: int, step: int) -> List[int]:
    a = np.array(sorted(set(map(float, sorted_values))))
    if a.size == 0 or k <= 0:
        return []
    i = int(np.argmin(np.abs(a - center_value)))
    picks = [i]
    L = i - step
    R = i + step
    while len(picks) < k and (L >= 0 or R < len(a)):
        if R < len(a):
            picks.append(R)
            if len(picks) >= k:
                break
        if L >= 0:
            picks.append(L)
        L -= step
        R += step
    return sorted(set(picks))[:k]

def _select_for_expiry(sub: pd.DataFrame, spot: float, k_per_expiry: int, step: int) -> List[str]:
    """
    Select ~k_per_expiry total (split across calls/puts) for a single expiry.
    """
    if sub.empty:
        return []
    calls = sub[sub["type"] == "call"].sort_values("strike")
    puts  = sub[sub["type"] == "put" ].sort_values("strike")
    k_each = max(1, k_per_expiry // 2)
    call_idxs = nearest_indices_every_other(calls["strike"].to_list(), spot, k_each, step)
    put_idxs  = nearest_indices_every_other(puts["strike"].to_list(),  spot, k_each, step)
    sel = []
    sel.extend(calls.iloc[call_idxs]["symbol"].tolist())
    sel.extend(puts.iloc[put_idxs]["symbol"].tolist())
    return sel

def select_symbols_both_sides(
    cdf: pd.DataFrame,
    spot: float,
    n_expiries: int,
    k_per_expiry: int,
    step: int,
    cap: int = 100,
) -> List[str]:
    """
    Original helper (kept for backward compatibility). Does NOT interleave.
    """
    exps = sorted(cdf["expiration"].unique())[:n_expiries]
    cdf = cdf[cdf["expiration"].isin(exps)].copy()
    k_each = max(1, k_per_expiry // 2)
    out: List[str] = []
    for exp in exps:
        sub = cdf[cdf["expiration"] == exp]
        calls = sub[sub["type"] == "call"].sort_values("strike")
        puts  = sub[sub["type"] == "put" ].sort_values("strike")
        c_idx = nearest_indices_every_other(calls["strike"].tolist(), spot, k_each, step)
        p_idx = nearest_indices_every_other(puts["strike"].tolist(),  spot, k_each, step)
        out.extend(calls.iloc[c_idx]["symbol"].tolist())
        out.extend(puts.iloc[p_idx]["symbol"].tolist())
        if len(out) >= cap:
            break
    return out[:cap]

def select_symbols_balanced(
    cdf: pd.DataFrame,
    spot: float,
    n_expiries: int,
    k_per_expiry: int,
    step: int,
    cap: int = 100,
    per_expiry_cap: int = 10,
) -> List[str]:
    """
    New helper:
      1) For each of the first n_expiries, pick near-ATM calls & puts (~k_per_expiry total).
      2) Enforce per-expiry cap (e.g., 10).
      3) Round-robin across expiries so the first `cap` symbols are spread out.
    This ensures the first 100 we send to Alpaca cover multiple expiries.
    """
    exps = sorted(cdf["expiration"].unique())[:n_expiries]
    cdf = cdf[cdf["expiration"].isin(exps)].copy()

    # build a list of symbols per expiry
    per_exp: Dict[pd.Timestamp, List[str]] = {}
    for exp in exps:
        sub = cdf[cdf["expiration"] == exp]
        sel = _select_for_expiry(sub, spot, k_per_expiry, step)
        if per_expiry_cap is not None and per_expiry_cap > 0:
            sel = sel[:per_expiry_cap]
        per_exp[exp] = sel

    # round-robin interleave
    out: List[str] = []
    # create iterators
    queues: List[List[str]] = [per_exp[e][:] for e in exps]
    exhausted = False
    while len(out) < cap and not exhausted:
        exhausted = True
        for q in queues:
            if q and len(out) < cap:
                out.append(q.pop(0))
                exhausted = False

    return out
