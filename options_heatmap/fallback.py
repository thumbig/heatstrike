from typing import Iterable, List, Optional
import pandas as pd
from .occ import parse_occ_symbol
from .snapshots import _years_until, _bs_gamma  # reuse the internal helpers

def gamma_from_symbols(
    symbols: Iterable[str],
    *,
    spot: float,
    risk_free_rate: float = 0.02,
    default_iv: Optional[float] = None,
    now_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build a gamma DataFrame (symbol, strike, expiration, gamma, type) from OCC symbols only,
    using default_iv for BS gamma. Symbols that can't be parsed are skipped.
    """
    rows: List[dict] = []
    now_ts = now_ts or pd.Timestamp.utcnow()
    if default_iv is None:
        return pd.DataFrame(columns=["symbol", "strike", "expiration", "gamma", "type"])

    for sym in symbols or []:
        strike, exp_dt, otype = parse_occ_symbol(sym)
        if strike is None or exp_dt is None:
            continue
        T = _years_until(pd.Timestamp(exp_dt), now_ts=now_ts)
        if T is None:
            continue
        gamma = _bs_gamma(spot, float(strike), T, risk_free_rate, float(default_iv))
        if gamma is None:
            continue
        rows.append({
            "symbol": sym,
            "strike": float(strike),
            "expiration": pd.to_datetime(exp_dt),
            "gamma": float(gamma),
            "type": otype,
        })

    return pd.DataFrame(rows, columns=["symbol", "strike", "expiration", "gamma", "type"])
