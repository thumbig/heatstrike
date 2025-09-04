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
)
from options_heatmap.selection import select_symbols_balanced
from options_heatmap.fetch import fetch_contracts_balanced
from options_heatmap.pricing import build_pricing_df
from options_heatmap.plot import plot_mispriced_heatmap

# ---------------------- Config ----------------------
UNDERLYING = os.getenv("UNDERLYING_SYMBOL", "AAPL")
N_EXPIRIES = int(os.getenv("N_EXPIRIES", "9"))
K_PER_EXPIRY = int(os.getenv("K_PER_EXPIRY", "11"))
EVERY_OTHER_STEP = int(os.getenv("EVERY_OTHER_STEP", "2"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02"))
OUTPUT_PATH = os.getenv("MISPRICED_HEATMAP_PATH", "mispriced_heatmap.png")
DEFAULT_IV = os.getenv("DEFAULT_IV")  # e.g., "0.22"
SHARE_IV_ACROSS_EXP = os.getenv("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False")
PER_EXPIRY_CAP = int(os.getenv("PER_EXPIRY_CAP", "10"))
THEO_IV_MODE = os.getenv("THEO_IV_MODE", "per_expiry_mean")  # "per_expiry_mean" | "snapshot_iv" | "default"
USE_PCT = os.getenv("MISPRICE_USE_PCT", "0") in ("1", "true", "True")

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
    # Use your gvars/pvars layout if not provided
    if trade is None or opt_data is None or stk_data is None:
        if paper:
            from gvars import ALPACA_API_KEY_2 as API_KEY, ALPACA_SECRET_KEY_2 as API_SECRET
            trade = trade or TradingClient(API_KEY, API_SECRET, paper=True)
            opt_data = opt_data or OptionHistoricalDataClient(API_KEY, API_SECRET)
            stk_data = stk_data or StockHistoricalDataClient(API_KEY, API_SECRET)
        else:
            from pvars import ALPACA_API_KEY as API_KEY, ALPACA_SECRET_KEY as API_SECRET
            trade = trade or TradingClient(API_KEY, API_SECRET, paper=False)
            opt_data = opt_data or OptionHistoricalDataClient(API_KEY, API_SECRET)
            stk_data = stk_data or StockHistoricalDataClient(API_KEY, API_SECRET)

    spot = _get_spot(stk_data, UNDERLYING)

    today = datetime.utcnow().date()
    base_req = GetOptionContractsRequest(
        underlying_symbols=[UNDERLYING],
        status="active",
        expiration_date_gte=today,
        expiration_date_lte=today + timedelta(days=365),
    )

    # Page + balance expiries on fetch
    raw_contracts = fetch_contracts_balanced(
        trade_client=trade,
        base_req=base_req,
        n_expiries=N_EXPIRIES,
        per_expiry_cap=PER_EXPIRY_CAP,
        max_pages=50,
        page_limit=100,
    )

    cdf = build_contracts_df(raw_contracts)
    if cdf.empty:
        raise RuntimeError("No contracts parsed (even after OCC fallback).")

    # Interleave per-expiry symbols (near-ATM) before snapshot
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

    snaps = opt_data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=selected_symbols))

    pdf = build_pricing_df(
        snaps,
        spot=spot,
        risk_free_rate=RISK_FREE_RATE,
        default_iv=float(DEFAULT_IV) if DEFAULT_IV not in (None, "") else None,
        share_iv_across_exp=SHARE_IV_ACROSS_EXP,
        theo_iv_mode=THEO_IV_MODE,
    )
    if pdf.empty:
        raise RuntimeError("No prices/IVs available to compute mispricing.")

    # quick visibility: expiries present
    try:
        by_exp = pdf.groupby(pd.to_datetime(pdf["expiration"]).dt.date).size()
        print(by_exp)
    except Exception:
        pass

    fig_or_ax = plot_mispriced_heatmap(
        pdf,
        title=f"Mispriced Options — {UNDERLYING}",
        save_path=OUTPUT_PATH,
        side_by_side=True,
        annotate="theo",        # print theoretical price in the cells
        annotate_fmt=".2f",     # show as 0.00 (currency-like)
        use_pct=USE_PCT,        # set MISPRICE_USE_PCT=1 to color by %
    )
    if fig_or_ax:
        print(f"Saved mispriced heatmap to: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()
    import matplotlib.pyplot as plt
    plt.show()
