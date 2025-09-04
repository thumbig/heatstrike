from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .occ import parse_occ_symbol
from .iv import bs_price as _bs_price, implied_vol_from_price as _iv_from_price

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

def build_pricing_df(
    snaps: Dict[str, Any],
    *,
    spot: float,
    risk_free_rate: float = 0.02,
    now_ts: Optional[pd.Timestamp] = None,
    default_iv: Optional[float] = None,
    share_iv_across_exp: bool = True,
    theo_iv_mode: str = "per_expiry_mean",  # "per_expiry_mean" | "snapshot_iv" | "default"
) -> pd.DataFrame:
    """
    Produce a pricing table with:
      symbol, type, strike, expiration, market_price, theo_price, mispricing, mispricing_pct, iv_market, iv_theo

    - market_price: latest trade, else quote mid.
    - iv_market: IV solved from market_price when greeks/IV were not provided by feed.
    - theo_price: Black–Scholes price using iv_theo, where:
        * per_expiry_mean: mean iv_market within that expiration (robust to noise)
        * snapshot_iv: snapshot IV if available, else iv_market, else default_iv
        * default: default_iv only (if absent → row will lack theo)
    - mispricing = market_price - theo_price
    - mispricing_pct = mispricing / theo_price (if theo_price > 0)
    """
    now_ts = now_ts or pd.Timestamp.utcnow()
    rows: List[dict] = []

    # First pass: parse and gather basics, market prices, and a candidate "market IV" (from snapshot or inversion)
    per_exp_ivs: Dict[pd.Timestamp, list] = {}
    staged: List[dict] = []

    for sym, snap in (snaps or {}).items():
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

        T = _years_until(exp, now_ts=now_ts)
        if T is None:
            continue

        mkt_px = _best_option_price(snap)
        if mkt_px is None or mkt_px <= 0:
            continue

        # Try to pull IV from snapshot; else invert from price
        iv_market = None
        greeks = _get(snap, "greeks")
        if greeks is not None:
            iv_market = _clean_num(_get(greeks, "implied_volatility"))
        iv_market = _clean_num(_get(snap, "implied_volatility")) or iv_market
        if iv_market is None:
            iv_market = _clean_num(_get(snap, "iv")) or _clean_num(_get(_get(snap, "metrics") or {}, "iv"))

        if iv_market is None:
            iv_market = _iv_from_price(
                price=mkt_px, S=spot, K=strike, T=T, r=risk_free_rate,
                opt_type=("call" if otype == "call" else "put")
            )

        staged.append({
            "symbol": sym,
            "type": otype,
            "strike": float(strike),
            "expiration": pd.to_datetime(exp),
            "T": T,
            "market_price": mkt_px,
            "iv_market": _clean_num(iv_market),
        })

        key = pd.to_datetime(exp).normalize()
        if _clean_num(iv_market) is not None:
            per_exp_ivs.setdefault(key, []).append(float(iv_market))

    # Build per-expiry IV (mean) if requested
    per_exp_mean_iv: Dict[pd.Timestamp, Optional[float]] = {}
    if theo_iv_mode == "per_expiry_mean":
        for k, lst in per_exp_ivs.items():
            per_exp_mean_iv[k] = float(np.mean(lst)) if lst else (float(default_iv) if default_iv is not None else None)

    # Second pass: compute theo IV/price and mispricing
    for it in staged:
        exp_key = pd.to_datetime(it["expiration"]).normalize()
        iv_theo = None
        if theo_iv_mode == "per_expiry_mean":
            iv_theo = per_exp_mean_iv.get(exp_key)
        elif theo_iv_mode == "snapshot_iv":
            iv_theo = it["iv_market"] if it["iv_market"] is not None else (float(default_iv) if default_iv is not None else None)
        elif theo_iv_mode == "default":
            iv_theo = float(default_iv) if default_iv is not None else None

        theo_price = None
        if iv_theo is not None and iv_theo > 0:
            theo_price = _bs_price(
                S=spot, K=it["strike"], T=it["T"], r=risk_free_rate,
                sigma=iv_theo, opt_type=("call" if it["type"] == "call" else "put")
            )

        mispricing = None
        mispricing_pct = None
        if theo_price is not None and theo_price > 0:
            mispricing = it["market_price"] - theo_price
            mispricing_pct = mispricing / theo_price

        rows.append({
            "symbol": it["symbol"],
            "type": it["type"],
            "strike": it["strike"],
            "expiration": it["expiration"],
            "market_price": it["market_price"],
            "iv_market": it["iv_market"],
            "iv_theo": _clean_num(iv_theo),
            "theo_price": _clean_num(theo_price),
            "mispricing": _clean_num(mispricing),
            "mispricing_pct": _clean_num(mispricing_pct),
        })

    return pd.DataFrame(rows, columns=[
        "symbol","type","strike","expiration",
        "market_price","iv_market","iv_theo","theo_price","mispricing","mispricing_pct"
    ])
