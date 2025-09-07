from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
import os
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest

from options_heatmap.fetch import fetch_contracts_balanced
from options_heatmap.selection import select_symbols_balanced
from options_heatmap import build_contracts_df
from options_heatmap.snapshots import build_gamma_df
from options_heatmap.pricing import build_pricing_df
from options_heatmap.filters import filter_contracts_df, pick_strikes_centered_around_atm


@dataclass
class Clients:
    trade: TradingClient
    opt_data: OptionHistoricalDataClient
    stk_data: StockHistoricalDataClient


def _get_env_key(names: list[str]) -> Optional[str]:
    for n in names:
        val = os.environ.get(n)
        if val:
            return val
    return None


def get_clients(*, paper: bool = True) -> Clients:
    api_key = _get_env_key(["ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_API_KEY_ID"])
    api_secret = _get_env_key(["ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"])
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca API creds in environment.")

    trade = TradingClient(api_key, api_secret, paper=paper)
    opt_data = OptionHistoricalDataClient(api_key, api_secret)
    stk_data = StockHistoricalDataClient(api_key, api_secret)
    return Clients(trade=trade, opt_data=opt_data, stk_data=stk_data)


def get_spot(stk_data: StockHistoricalDataClient, symbol: str) -> float:
    lt = stk_data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    trade_obj = lt.get(symbol) if isinstance(lt, dict) else lt
    spot = getattr(trade_obj, "price", None)
    if spot is None and hasattr(trade_obj, "trade") and hasattr(trade_obj.trade, "price"):
        spot = trade_obj.trade.price
    if spot is None:
        raise RuntimeError(f"Couldn't determine spot price for {symbol}.")
    return float(spot)


def fetch_contracts(trade: TradingClient, underlying: str,
                    *, n_expiries: int, per_expiry_cap: int) -> pd.DataFrame:
    today = datetime.utcnow().date()
    base_req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status="active",
        expiration_date_gte=today,
        expiration_date_lte=today + timedelta(days=365),
    )
    raw_contracts = fetch_contracts_balanced(
        trade_client=trade,
        base_req=base_req,
        n_expiries=n_expiries,
        per_expiry_cap=per_expiry_cap,
        max_pages=50,
        page_limit=100,
    )
    cdf = build_contracts_df(raw_contracts)
    if cdf.empty:
        raise RuntimeError("No contracts parsed (even after OCC fallback).")
    return cdf


def select_symbols(
    cdf: pd.DataFrame,
    *,
    spot: float,
    n_expiries: int,
    k_per_expiry: int,
    step: int,
    per_expiry_cap: int,
    # NEW optional constraints
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    exp_start: Optional[pd.Timestamp] = None,
    exp_end: Optional[pd.Timestamp] = None,
    strikes_count: Optional[int] = None,
) -> List[str]:
    """
    Apply optional strike/expiry filters (and/or a centered strike-count window),
    then perform balanced selection across expiries.
    """
    work = cdf

    # Filter by strike / expiry ranges if provided
    if any(v is not None for v in (strike_min, strike_max, exp_start, exp_end)):
        work = filter_contracts_df(
            work,
            strike_min=strike_min,
            strike_max=strike_max,
            exp_start=exp_start,
            exp_end=exp_end,
        )

    # If strikes_count is provided, narrow strikes per-expiry to a centered window around ATM
    if strikes_count and strikes_count > 0:
        # compute per-expiry strike windows to keep shape roughly balanced
        keep_rows = []
        for exp, grp in work.groupby("expiration"):
            strikes = grp["strike"].unique()
            chosen = pick_strikes_centered_around_atm(strikes, spot=spot, count=strikes_count)
            keep_rows.append(grp[grp["strike"].isin(chosen)])
        work = pd.concat(keep_rows, ignore_index=True) if keep_rows else work

    symbols = select_symbols_balanced(
        cdf=work,
        spot=spot,
        n_expiries=n_expiries,
        k_per_expiry=k_per_expiry,
        step=step,
        cap=100,
        per_expiry_cap=per_expiry_cap,
    )
    if not symbols:
        raise RuntimeError("Symbol selection produced empty set after applying constraints.")
    return symbols


def snapshot(opt_data: OptionHistoricalDataClient, symbols: List[str]):
    return opt_data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols))


def prepare_df(metric: str,
               snaps,
               *,
               spot: float,
               risk_free_rate: float = 0.02,
               default_iv: Optional[float] = None,
               share_iv_across_exp: bool = True,
               theo_iv_mode: str = "per_expiry_mean") -> pd.DataFrame:
    m = metric.lower()
    if m == "mispriced":
        df = build_pricing_df(
            snaps, spot=spot, risk_free_rate=risk_free_rate,
            default_iv=default_iv, share_iv_across_exp=share_iv_across_exp,
            theo_iv_mode=theo_iv_mode,
        )
        if df.empty:
            raise RuntimeError("No prices/IVs available to compute mispricing.")
        return df

    df = build_gamma_df(
        snaps, spot=spot, risk_free_rate=risk_free_rate,
        default_iv=default_iv, share_iv_across_exp=share_iv_across_exp,
    )
    if df.empty:
        raise RuntimeError("No greeks available (even after fallback).")
    return df


def nearest_atm_strike(df: pd.DataFrame, spot: float) -> Optional[float]:
    try:
        strikes = pd.Series(sorted(df["strike"].unique()))
        return float(strikes.iloc[(strikes - spot).abs().argmin()])
    except Exception:
        return None


