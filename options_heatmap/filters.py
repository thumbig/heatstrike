from __future__ import annotations
from typing import Optional, Sequence, List
import pandas as pd
import numpy as np

def filter_contracts_df(
    cdf: pd.DataFrame,
    *,
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    exp_start: Optional[pd.Timestamp] = None,
    exp_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if cdf is None or cdf.empty:
        return cdf

    out = cdf.copy()
    if strike_min is not None:
        out = out[out["strike"] >= float(strike_min)]
    if strike_max is not None:
        out = out[out["strike"] <= float(strike_max)]

    if exp_start is not None:
        exp_start = pd.to_datetime(exp_start)
        out = out[pd.to_datetime(out["expiration"]) >= exp_start]
    if exp_end is not None:
        exp_end = pd.to_datetime(exp_end)
        out = out[pd.to_datetime(out["expiration"]) <= exp_end]

    return out


def pick_strikes_centered_around_atm(
    strikes: Sequence[float] | np.ndarray,
    *,
    spot: float,
    count: int,
) -> List[float]:
    """
    Select ~count distinct strikes centered around ATM (closest to spot),
    balanced approximately on each side. If count is odd, includes ATM.
    """
    # FIX: don't use `if not strikes` on numpy arrays (ambiguous truth value)
    if strikes is None:
        return []
    try:
        n = len(strikes)
    except TypeError:
        return []
    if n == 0 or count <= 0:
        return []

    uniq = np.unique(np.asarray(strikes, dtype=float))
    atm_idx = int(np.abs(uniq - spot).argmin())

    if count == 1:
        return [float(uniq[atm_idx])]

    selected = [atm_idx]
    left = atm_idx - 1
    right = atm_idx + 1
    while len(selected) < count and (left >= 0 or right < len(uniq)):
        if len(selected) % 2 == 1:
            if right < len(uniq):
                selected.append(right); right += 1
            elif left >= 0:
                selected.append(left); left -= 1
        else:
            if left >= 0:
                selected.append(left); left -= 1
            elif right < len(uniq):
                selected.append(right); right += 1

    selected = sorted(set(selected))
    return [float(uniq[i]) for i in selected]

