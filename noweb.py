#!/usr/bin/env python3
# Force a headless backend so Qt/Wayland isn’t required
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path
import pandas as pd

from options_heatmap.core import (
    get_clients, get_spot, fetch_contracts, select_symbols, snapshot,
    prepare_df, nearest_atm_strike
)
from options_heatmap.plot import (
    plot_gamma_heatmap, plot_theta_heatmap, plot_mispriced_heatmap
)

def _parse_date_or_none(s: str | None):
    if not s:
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(description="Standalone Seaborn options heatmaps")
    p.add_argument("-m","--metric", choices=["mispriced","gamma","theta"], default="mispriced")
    p.add_argument("-s","--symbol", required=True, help="Underlying symbol (e.g., AAPL, SPY, NVDA, TSLA)")
    p.add_argument("--paper", action="store_true", help="Use paper account creds (env vars)")
    p.add_argument("--n-expiries", type=int, default=9)
    p.add_argument("--k-per-expiry", type=int, default=11)
    p.add_argument("--step", type=int, default=2, help="Every Nth strike from ATM selection")
    p.add_argument("--per-expiry-cap", type=int, default=10)
    p.add_argument("--rfr", type=float, default=0.02, help="Risk-free rate")
    p.add_argument("--default-iv", type=float, default=None)
    p.add_argument("--share-iv", action="store_true", help="Share fallback IV per-expiry")
    p.add_argument("--theo-iv-mode", choices=["per_expiry_mean","snapshot_iv","default"], default="per_expiry_mean")
    p.add_argument("--side-by-side", action="store_true", help="Plot Calls and Puts side-by-side")
    p.add_argument("--annotate", choices=["iv","symbol","theo","none"], default="iv",
                   help="Cell text: mispriced→'theo' recommended; gamma/theta→'iv' or 'symbol'")
    p.add_argument("--y-order", choices=["desc","asc"], default="desc", help="Strike axis order")
    p.add_argument("--save", type=str, default=None, help="Path to save PNG")
    p.add_argument("--title", type=str, default=None)
    # NEW constraints
    p.add_argument("--strike-min", type=float, default=None, help="Minimum strike to include")
    p.add_argument("--strike-max", type=float, default=None, help="Maximum strike to include")
    p.add_argument("--exp-start", type=str, default=None, help="Earliest expiration (YYYY-MM-DD)")
    p.add_argument("--exp-end", type=str, default=None, help="Latest expiration (YYYY-MM-DD)")
    p.add_argument("--strikes-count", type=int, default=None, help="Centered count around ATM per-expiry")
    args = p.parse_args()

    annotate = None if args.annotate == "none" else args.annotate
    y_desc = (args.y_order == "desc")

    exp_start = _parse_date_or_none(args.exp_start)
    exp_end = _parse_date_or_none(args.exp_end)

    clients = get_clients(paper=args.paper)
    spot = get_spot(clients.stk_data, args.symbol)
    cdf = fetch_contracts(clients.trade, args.symbol, n_expiries=args.n_expiries, per_expiry_cap=args.per_expiry_cap)

    syms = select_symbols(
        cdf,
        spot=spot,
        n_expiries=args.n_expiries,
        k_per_expiry=args.k_per_expiry,
        step=args.step,
        per_expiry_cap=args.per_expiry_cap,
        strike_min=args.strike_min,
        strike_max=args.strike_max,
        exp_start=exp_start,
        exp_end=exp_end,
        strikes_count=args.strikes_count,
    )

    snaps = snapshot(clients.opt_data, syms)
    df = prepare_df(
        args.metric, snaps, spot=spot, risk_free_rate=args.rfr,
        default_iv=args.default_iv, share_iv_across_exp=args.share_iv, theo_iv_mode=args.theo_iv_mode
    )
    atm = nearest_atm_strike(df, spot)
    title = args.title or f"{args.metric.title()} — {args.symbol} (Spot ${spot:.2f})"
    save_path = args.save

    if args.metric == "mispriced":
        fig_or_ax = plot_mispriced_heatmap(
            df, title=title, save_path=save_path, side_by_side=args.side_by_side,
            annotate=("theo" if annotate is None else annotate), annotate_fmt=".2f",
            use_pct=False, atm_strike=atm, y_descending=y_desc
        )
    elif args.metric == "gamma":
        fig_or_ax = plot_gamma_heatmap(
            df, title=title, save_path=save_path, side_by_side=args.side_by_side,
            annotate=("iv" if annotate is None else annotate), annotate_fmt=".1f",
            atm_strike=atm, y_descending=y_desc
        )
    else:
        fig_or_ax = plot_theta_heatmap(
            df, title=title, save_path=save_path, side_by_side=args.side_by_side,
            annotate=("iv" if annotate is None else annotate), annotate_fmt=".1f",
            atm_strike=atm, y_descending=y_desc
        )

    if save_path:
        print(f"Saved figure to: {Path(save_path).resolve()}")

if __name__ == "__main__":
    main()


