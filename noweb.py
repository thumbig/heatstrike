import os
from datetime import datetime, timedelta

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest

from options_heatmap import (
    normalize_contracts_response,
    build_contracts_df,
    select_symbols_both_sides,  # kept for compatibility
    build_gamma_df,
    plot_gamma_heatmap,
)
from options_heatmap.selection import select_symbols_balanced
from options_heatmap.fetch import fetch_contracts_balanced

# ---------------------- Config ----------------------
UNDERLYING = os.getenv("UNDERLYING_SYMBOL", "AAPL")
N_EXPIRIES = int(os.getenv("N_EXPIRIES", "9"))
K_PER_EXPIRY = int(os.getenv("K_PER_EXPIRY", "11"))
EVERY_OTHER_STEP = int(os.getenv("EVERY_OTHER_STEP", "2"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02"))
OUTPUT_PATH = os.getenv("HEATMAP_PATH", "gamma_heatmap.png")
DEFAULT_IV = os.getenv("DEFAULT_IV")  # e.g., "0.22"
SHARE_IV_ACROSS_EXP = os.getenv("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False")
PER_EXPIRY_CAP = int(os.getenv("PER_EXPIRY_CAP", "10"))

def _get_spot(stk_data: StockHistoricalDataClient, symbol: str) -> float:
    lt = stk_data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    trade_obj = lt.get(symbol) if isinstance(lt, dict) else lt
    spot = getattr(trade_obj, "price", None)
    if spot is None and hasattr(trade_obj, "trade") and hasattr(trade_obj.trade, "price"):
        spot = trade_obj.trade.price
    if spot is None:
        raise RuntimeError("Couldn't determine spot price from latest trade.")
    return float(spot)

def main(trade: TradingClient | None = None,
         opt_data: OptionHistoricalDataClient | None = None,
         stk_data: StockHistoricalDataClient | None = None,
         paper: bool = True) -> None:
    if trade is None or opt_data is None or stk_data is None:
        ALPACA_API_KEY_2 = os.getenv("ALPACA_API_KEY_2", "anonymous")
        ALPACA_SECRET_KEY_2 = os.getenv("ALPACA_SECRET_KEY_2", "anonymous")
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "anonymous")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "anonymous")

    if paper:
        trade = trade or TradingClient(ALPACA_API_KEY_2, ALPACA_SECRET_KEY_2, paper=True)
        opt_data = opt_data or OptionHistoricalDataClient(ALPACA_API_KEY_2, ALPACA_SECRET_KEY_2)
        stk_data = stk_data or StockHistoricalDataClient(ALPACA_API_KEY_2, ALPACA_SECRET_KEY_2)
    else:
        trade = trade or TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=False)
        opt_data = opt_data or OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        stk_data = stk_data or StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    spot = _get_spot(stk_data, UNDERLYING)

    today = datetime.utcnow().date()
    base_req = GetOptionContractsRequest(
        underlying_symbols=[UNDERLYING],
        status="active",
        expiration_date_gte=today,
        expiration_date_lte=today + timedelta(days=365),
    )

    # >>> NEW: fetch contracts paged & balanced across expiries (per-expiry cap honored here)
    raw_contracts = fetch_contracts_balanced(
        trade_client=trade,
        base_req=base_req,
        n_expiries=N_EXPIRIES,
        per_expiry_cap=PER_EXPIRY_CAP,
        max_pages=50,
        page_limit=100,  # typical Alpaca page size; adjust if your SDK differs
    )

    cdf = build_contracts_df(raw_contracts)
    if cdf.empty:
        raise RuntimeError("No contracts parsed (even after OCC fallback).")

    # Within those balanced contracts, still pick near-ATM calls & puts (keeps behavior stable)
    selected_symbols = select_symbols_balanced(
        cdf=cdf,
        spot=spot,
        n_expiries=N_EXPIRIES,
        k_per_expiry=K_PER_EXPIRY,
        step=EVERY_OTHER_STEP,
        cap=100,
        per_expiry_cap=PER_EXPIRY_CAP,
    )
    if not selected_symbols:
        raise RuntimeError("Symbol selection produced empty set.")

    # One snapshot request for the interleaved symbols
    snaps = opt_data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=selected_symbols))

    gamma_df = build_gamma_df(
        snaps,
        spot=spot,
        risk_free_rate=RISK_FREE_RATE,
        default_iv=float(DEFAULT_IV) if DEFAULT_IV not in (None, "") else None,
        share_iv_across_exp=SHARE_IV_ACROSS_EXP,
    )

    if gamma_df.empty:
        raise RuntimeError("No gamma available for selected symbols (even after fallback).")

    # quick visibility: how many per expiry made it through
    try:
        by_exp = gamma_df.groupby(pd.to_datetime(gamma_df["expiration"]).dt.date).size()
        print(by_exp)
    except Exception:
        pass

    title = f"Gamma Heatmap for {UNDERLYING}"
    fig_or_ax = plot_gamma_heatmap(
        gamma_df,
        title=title,
        save_path=OUTPUT_PATH,
        side_by_side=True,
        annotate="iv",  # or "symbol", "iv" or None
        annotate_fmt=".1f",  # used for numeric annotations like IV%
    )
    if fig_or_ax:
        print(f"Saved heatmap to: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main(paper=True)
    import matplotlib.pyplot as plt
    plt.show()
