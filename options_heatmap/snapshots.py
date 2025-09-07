from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .occ import parse_occ_symbol
from .iv import bs_gamma as _bs_gamma, bs_theta as _bs_theta, implied_vol_from_price as _iv_from_price

def _get(obj: Any, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts

def _years_until(exp_ts: pd.Timestamp, now_ts: Optional[pd.Timestamp] = None) -> Optional[float]:
    if exp_ts is None:
        return None
    now_ts = now_ts or pd.Timestamp.utcnow()
    exp_n = _to_naive_utc(pd.Timestamp(exp_ts))
    now_n = _to_naive_utc(pd.Timestamp(now_ts))
    dt_days = (exp_n - now_n).total_seconds() / 86400.0
    T = dt_days / 365.0
    return T if T > 0 else None

def _clean_num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return x

def _best_option_price(snap: Any) -> Optional[float]:
    lt = _get(snap, "latest_trade")
    if lt is not None:
        p = _clean_num(_get(lt, "price"))
        if p is not None:
            return p
    lq = _get(snap, "latest_quote")
    if lq is not None:
        bid = _clean_num(_get(lq, "bid_price"))
        ask = _clean_num(_get(lq, "ask_price"))
        if bid is not None and ask is not None and ask > 0:
            mid = 0.5 * (bid + ask)
            return mid if mid > 0 else None
    return None

def build_gamma_df(
    snaps: Dict[str, Any],
    *,
    spot: Optional[float] = None,
    risk_free_rate: float = 0.02,
    now_ts: Optional[pd.Timestamp] = None,
    default_iv: Optional[float] = None,
    share_iv_across_exp: bool = True,
) -> pd.DataFrame:
    """
    Returns columns: symbol, strike, expiration, gamma, theta, type, iv.
    - gamma is used for color in your existing plotter.
    - theta is included for the theta plotter.
    - iv is the volatility actually used (from snapshot, default_iv, or inverted from price).
    """
    rows: List[dict] = []
    now_ts = now_ts or pd.Timestamp.utcnow()
    iv_by_exp: Dict[pd.Timestamp, float] = {}

    for sym, snap in (snaps or {}).items():
        # Parse OCC
        ps, pe, ptype = parse_occ_symbol(sym)
        strike = _clean_num(ps)
        exp = pd.to_datetime(pe) if pe is not None else None
        otype = ptype

        details = _get(snap, "details")
        if details is not None:
            strike = _clean_num(_get(details, "strike_price")) or strike
            exp = _get(details, "expiration_date") or exp
        if strike is None or exp is None:
            continue
        try:
            exp = pd.to_datetime(exp)
        except Exception:
            continue

        # Base greeks from snapshot
        gamma = None
        theta = None
        iv_used = None
        greeks = _get(snap, "greeks")
        if greeks is not None:
            gamma = _clean_num(_get(greeks, "gamma"))
            theta = _clean_num(_get(greeks, "theta"))
            iv_used = _clean_num(_get(greeks, "implied_volatility")) or iv_used

        # try other IV fields
        iv_used = _clean_num(_get(snap, "implied_volatility")) or iv_used
        if iv_used is None:
            iv_used = _clean_num(_get(snap, "iv")) or _clean_num(_get(_get(snap, "metrics") or {}, "iv"))

        # Fallbacks via BS if we have spot
        if (gamma is None or theta is None) and spot is not None:
            key = pd.to_datetime(exp).normalize()
            if iv_used is None and share_iv_across_exp and key in iv_by_exp:
                iv_used = iv_by_exp[key]
            if iv_used is None and default_iv is not None:
                iv_used = _clean_num(default_iv)

            T = _years_until(exp, now_ts=now_ts)

            if iv_used is None and T is not None:
                opt_px = _best_option_price(snap)
                if opt_px is not None:
                    iv_used = _iv_from_price(
                        price=opt_px, S=spot, K=strike, T=T, r=risk_free_rate,
                        opt_type=("call" if otype == "call" else "put")
                    )

            if (T is not None) and (iv_used is not None):
                if gamma is None:
                    gamma = _bs_gamma(spot, strike, T, risk_free_rate, iv_used)
                if theta is None:
                    theta = _bs_theta(spot, strike, T, risk_free_rate, iv_used, "call" if otype == "call" else "put")
                if share_iv_across_exp and iv_used is not None:
                    iv_by_exp[key] = iv_used

        if gamma is None and theta is None:
            # nothing to plot; skip
            continue

        rows.append({
            "symbol": sym,
            "strike": float(strike),
            "expiration": pd.to_datetime(exp),
            "gamma": float(gamma) if gamma is not None else np.nan,
            "theta": float(theta) if theta is not None else np.nan,
            "type": otype,
            "iv": _clean_num(iv_used),
        })

    return pd.DataFrame(rows, columns=["symbol", "strike", "expiration", "gamma", "theta", "type", "iv"])
