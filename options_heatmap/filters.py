from __future__ import annotations
from typing import Iterable, Optional, Sequence, List
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
    """
    Filter a contracts DataFrame by strike range and/or expiration range.
    Operates on the contracts-level DataFrame returned by build_contracts_df().
    Expects columns: 'strike', 'expiration'
    """
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
    strikes: Sequence[float],
    *,
    spot: float,
    count: int,
) -> List[float]:
    """
    Select ~count distinct strikes centered around ATM (closest to spot),
    balanced approximately on each side. If count is odd, includes ATM;
    if even, picks one more from the side with more availability.

    Strikes are assumed numeric (floats); returns a sorted list.
    """
    if not strikes or count <= 0:
        return []

    uniq = np.unique(np.asarray(strikes, dtype=float))
    # Find ATM index
    atm_idx = int(np.abs(uniq - spot).argmin())

    if count == 1:
        return [float(uniq[atm_idx])]

    # Build window expanding left/right
    selected = [atm_idx]
    left = atm_idx - 1
    right = atm_idx + 1
    while len(selected) < count and (left >= 0 or right < len(uniq)):
        # Alternate sides where possible, bias toward availability
        if len(selected) % 2 == 1:
            # prefer right first
            if right < len(uniq):
                selected.append(right)
                right += 1
            elif left >= 0:
                selected.append(left)
                left -= 1
        else:
            # then left
            if left >= 0:
                selected.append(left)
                left -= 1
            elif right < len(uniq):
                selected.append(right)
                right += 1

    selected = sorted(set(selected))
    return list(map(lambda i: float(uniq[i]), selected))


