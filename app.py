import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request, send_from_directory, flash

# Alpaca + your modules
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest

from options_heatmap.fetch import fetch_contracts_balanced
from options_heatmap.selection import select_symbols_balanced
from options_heatmap import build_contracts_df
from options_heatmap.snapshots import build_gamma_df
from options_heatmap.pricing import build_pricing_df
from options_heatmap.plot import (
    plot_gamma_heatmap,
    plot_theta_heatmap,
    plot_mispriced_heatmap,
)

# ---------------------- Config ----------------------
# You can also adapt to gvars/pvars; this file uses env vars by default.
UNDERLYING_DEFAULT = os.getenv("UNDERLYING_SYMBOL", "AAPL")
OUTPUT_DIR = Path(os.getenv("STATIC_DIR", "static"))
OUTPUT_DIR.mkdir(exist_ok=True)

N_EXPIRIES = int(os.getenv("N_EXPIRIES", "9"))
K_PER_EXPIRY = int(os.getenv("K_PER_EXPIRY", "11"))
EVERY_OTHER_STEP = int(os.getenv("EVERY_OTHER_STEP", "2"))
PER_EXPIRY_CAP = int(os.getenv("PER_EXPIRY_CAP", "10"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02"))

DEFAULT_IV = os.getenv("DEFAULT_IV")  # e.g. "0.22"
DEFAULT_IV_F = float(DEFAULT_IV) if DEFAULT_IV not in (None, "") else None
SHARE_IV_ACROSS_EXP = os.getenv("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False")
THEO_IV_MODE = os.getenv("THEO_IV_MODE", "per_expiry_mean")  # for mispriced view

ALLOWED_UNDERLYINGS = ["SPY", "AAPL", "TSLA", "NVDA", "SPX", "QQQ", "IWM"]

# ---------------------- App ----------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "Koughai8")

def _get_clients(paper: bool = True):
    if paper:
        API_KEY = os.getenv("ALPACA_API_KEY_2", "anonymous")
        API_SECRET = os.getenv("ALPACA_SECRET_KEY_2", "anonymous")
    else:
        API_KEY = os.getenv("ALPACA_API_KEY", "anonymous")
        API_SECRET = os.getenv("ALPACA_SECRET_KEY", "anonymous")

    if not API_KEY or not API_SECRET:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")

    trade = TradingClient(API_KEY, API_SECRET, paper=paper)
    opt_data = OptionHistoricalDataClient(API_KEY, API_SECRET)
    stk_data = StockHistoricalDataClient(API_KEY, API_SECRET)
    return trade, opt_data, stk_data

def _get_spot(stk_data: StockHistoricalDataClient, symbol: str) -> float:
    lt = stk_data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    trade_obj = lt.get(symbol) if isinstance(lt, dict) else lt
    spot = getattr(trade_obj, "price", None)
    if spot is None and hasattr(trade_obj, "trade") and hasattr(trade_obj.trade, "price"):
        spot = trade_obj.trade.price
    if spot is None:
        raise RuntimeError("Couldn't determine spot price from latest trade.")
    return float(spot)

def _balanced_contracts(trade, underlying: str):
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
        n_expiries=N_EXPIRIES,
        per_expiry_cap=PER_EXPIRY_CAP,
        max_pages=50,
        page_limit=100,
    )
    cdf = build_contracts_df(raw_contracts)
    if cdf.empty:
        raise RuntimeError("No contracts parsed (even after OCC fallback).")
    return cdf

def _select_symbols(cdf: pd.DataFrame, spot: float):
    symbols = select_symbols_balanced(
        cdf=cdf,
        spot=spot,
        n_expiries=N_EXPIRIES,
        k_per_expiry=K_PER_EXPIRY,
        step=EVERY_OTHER_STEP,
        cap=100,
        per_expiry_cap=PER_EXPIRY_CAP,
    )
    if not symbols:
        raise RuntimeError("Symbol selection produced empty set.")
    return symbols

def _snapshot(opt_data, symbols):
    return opt_data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbols))

def _prepare_df(metric: str, symbol: str, paper: bool):
    trade, opt_data, stk_data = _get_clients(paper=paper)
    spot = _get_spot(stk_data, symbol)
    cdf = _balanced_contracts(trade, symbol)
    syms = _select_symbols(cdf, spot)
    snaps = _snapshot(opt_data, syms)
    if metric == "mispriced":
        df = build_pricing_df(
            snaps,
            spot=spot,
            risk_free_rate=RISK_FREE_RATE,
            default_iv=DEFAULT_IV_F,
            share_iv_across_exp=SHARE_IV_ACROSS_EXP,
            theo_iv_mode=THEO_IV_MODE,
        )
        if df.empty:
            raise RuntimeError("No prices/IVs available to compute mispricing.")
    else:
        df = build_gamma_df(
            snaps,
            spot=spot,
            risk_free_rate=RISK_FREE_RATE,
            default_iv=DEFAULT_IV_F,
            share_iv_across_exp=SHARE_IV_ACROSS_EXP,
        )
        if df.empty:
            raise RuntimeError("No greeks available (even after fallback).")
    return df, spot

def _render_and_save(metric: str, underlying: str, paper: bool = True) -> Optional[str]:
    df, spot = _prepare_df(metric, underlying, paper)
    stamp = int(time.time())
    fname = f"heatmap_{metric}_{underlying}_{stamp}.png"
    fpath = OUTPUT_DIR / fname
    if metric == "mispriced":
        fig = plot_mispriced_heatmap(
            df, title=f"Mispriced — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="theo", annotate_fmt=".2f", use_pct=False,
        )
    elif metric == "gamma":
        fig = plot_gamma_heatmap(
            df, title=f"Gamma — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="iv", annotate_fmt=".1f",
        )
    else:
        fig = plot_theta_heatmap(
            df, title=f"Theta — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="iv", annotate_fmt=".1f",
        )
    if fig:
        return f"static/{fname}", spot
    return None, spot

@app.route("/", methods=["GET", "POST"])
def index():
    symbol = request.values.get("symbol", UNDERLYING_DEFAULT).upper()
    dropdown = request.values.get("dropdown_symbol", "")
    metric = request.values.get("metric", "mispriced").lower()
    paper = request.values.get("paper", "1") in ("1", "true", "True")

    typed = request.values.get("symbol", "").strip()
    if not typed and dropdown:
        symbol = dropdown.upper()

    if request.method == "POST":
        try:
            img_rel, spot = _render_and_save(metric, symbol, paper=paper)
            if not img_rel:
                flash("No figure produced.", "warning")
            return render_template(
                "index.html",
                allowed=ALLOWED_UNDERLYINGS,
                selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
                symbol=symbol, metric=metric, image_path=img_rel,
                download_href=img_rel, paper="1" if paper else "0",
                spot=spot,
            )
        except Exception as e:
            flash(str(e), "error")

    # initial GET: show live spot too
    spot_val = None
    try:
        _, _, stk = _get_clients(paper=True)
        spot_val = _get_spot(stk, symbol)
    except Exception:
        pass

    return render_template(
        "index.html",
        allowed=ALLOWED_UNDERLYINGS,
        selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
        symbol=symbol, metric=metric, image_path=None, download_href=None,
        paper="1" if (os.getenv("DEFAULT_PAPER", "1") in ("1", "true", "True")) else "0",
        spot=spot_val,
    )

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ---------------------- Interactive route (Plotly) ----------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _format_mmdd(seq):
    return [pd.to_datetime(c).strftime("%m/%d") for c in seq]

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    pv = df.pivot_table(index="strike", columns="expiration", values=value_col, aggfunc="mean")
    # Descending strikes for y-axis
    pv = pv.sort_index(axis=0, ascending=False)
    if len(pv.columns) > 0:
        pv = pv.reindex(sorted(pv.columns), axis=1)
    return pv

def _hover_text(df: pd.DataFrame, value_col: str, extra_cols: list) -> pd.DataFrame:
    """Build a pivoted text matrix matching value_col pivot, containing rich hover text."""
    if df.empty:
        return pd.DataFrame()
    cols = ["strike", "expiration", "symbol", value_col] + [c for c in extra_cols if c in df.columns]
    tmp = df[cols].copy()
    tmp["expiration"] = pd.to_datetime(tmp["expiration"])
    # We’ll fill a dict keyed by (strike, exp) with a multi-line hover string
    def row_text(r):
        bits = [f"<b>{r['symbol']}</b>",
                f"Strike: {r['strike']}",
                f"Exp: {pd.to_datetime(r['expiration']).strftime('%Y-%m-%d')}"]
        if value_col in r and pd.notna(r[value_col]):
            bits.append(f"{value_col.title()}: {r[value_col]:.4f}")
        for k in extra_cols:
            if k in r and pd.notna(r[k]):
                # Pretty-print a few common ones
                if k == "iv":
                    bits.append(f"IV: {r[k]*100:.2f}%")
                elif k in ("theo_price","market_price","mispricing"):
                    bits.append(f"{k.replace('_',' ').title()}: ${r[k]:.2f}")
                else:
                    bits.append(f"{k}: {r[k]}")
        return "<br>".join(bits)
    tmp["hover"] = tmp.apply(row_text, axis=1)
    # Pivot by taking the first hover text per (strike, exp)
    hov = tmp.pivot_table(index="strike", columns="expiration", values="hover", aggfunc="first")
    hov = hov.sort_index(axis=0, ascending=False)
    if len(hov.columns) > 0:
        hov = hov.reindex(sorted(hov.columns), axis=1)
    return hov

@app.route("/interactive", methods=["GET", "POST"])
def interactive():
    symbol = request.values.get("symbol", UNDERLYING_DEFAULT).upper()
    dropdown = request.values.get("dropdown_symbol", "")
    metric = request.values.get("metric", "mispriced").lower()
    paper = request.values.get("paper", "1") in ("1", "true", "True")

    typed = request.values.get("symbol", "").strip()
    if not typed and dropdown:
        symbol = dropdown.upper()

    try:
        df, spot = _prepare_df(metric, symbol, paper=paper)
    except Exception as e:
        flash(str(e), "error")
        return render_template(
            "interactive.html",
            allowed=ALLOWED_UNDERLYINGS,
            selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
            symbol=symbol, metric=metric, paper="1" if paper else "0",
            plot_html="<p>Error generating plot.</p>", spot=None,
        )

    value_col = {
        "mispriced": "mispricing",
        "gamma": "gamma",
        "theta": "theta",
    }[metric]

    # Split by calls/puts for side-by-side subplots
    has_type = "type" in df.columns
    calls = df[df["type"] == "call"] if has_type else df
    puts  = df[df["type"] == "put"]  if has_type else pd.DataFrame(columns=df.columns)

    pv_c = _pivot(calls, value_col)
    pv_p = _pivot(puts,  value_col)

    x_c = _format_mmdd(pv_c.columns) if not pv_c.empty else []
    x_p = _format_mmdd(pv_p.columns) if not pv_p.empty else []
    y_c = list(pv_c.index) if not pv_c.empty else []
    y_p = list(pv_p.index) if not pv_p.empty else y_c

    # Hover text matrices
    extra_cols = []
    if metric in ("gamma", "theta"):
        extra_cols = ["iv"]
    elif metric == "mispriced":
        extra_cols = ["theo_price", "market_price", "iv_theo", "iv_market", "mispricing_pct", "iv"]

    hov_c = _hover_text(calls, value_col, extra_cols) if not calls.empty else pd.DataFrame()
    hov_p = _hover_text(puts,  value_col, extra_cols) if not puts.empty else pd.DataFrame()
    text_c = hov_c.values if not hov_c.empty else None
    text_p = hov_p.values if not hov_p.empty else None

    # Colormaps: mispriced=bluered; gamma=coolwarm; theta=RdBu_r
    cmap = "bluered" if metric in ("mispriced","gamma") else "RdBu_r"
    zmid = 0

    if has_type:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Calls", "Puts"))
        if not pv_c.empty:
            fig.add_trace(
                go.Heatmap(z=pv_c.values, x=x_c, y=y_c, colorscale=cmap, zmid=zmid,
                           colorbar=dict(title=value_col.title()),
                           hoverinfo="text", text=text_c),
                row=1, col=1
            )
        if not pv_p.empty:
            fig.add_trace(
                go.Heatmap(z=pv_p.values, x=x_p, y=y_p, colorscale=cmap, zmid=zmid,
                           showscale=False, hoverinfo="text", text=text_p),
                row=1, col=2
            )
        fig.update_yaxes(title_text="Strike", row=1, col=1, autorange="reversed")  # reversed: highest at top
        fig.update_xaxes(title_text="Expiration (MM/DD)", row=1, col=1)
        fig.update_xaxes(title_text="Expiration (MM/DD)", row=1, col=2)
        fig.update_layout(title=f"{value_col.title()} — {symbol}  (Spot ${spot:.2f})",
                          margin=dict(l=70, r=20, t=60, b=60), height=600)
    else:
        fig = go.Figure(
            data=go.Heatmap(
                z=pv_c.values, x=x_c, y=y_c,
                colorscale=cmap, zmid=zmid,
                colorbar=dict(title=value_col.title()),
                hoverinfo="text", text=text_c
            )
        )
        fig.update_yaxes(title_text="Strike", autorange="reversed")
        fig.update_xaxes(title_text="Expiration (MM/DD)")
        fig.update_layout(title=f"{value_col.title()} — {symbol}  (Spot ${spot:.2f})",
                          margin=dict(l=70, r=20, t=60, b=60), height=600)

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return render_template(
        "interactive.html",
        allowed=ALLOWED_UNDERLYINGS,
        selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
        symbol=symbol, metric=metric, paper="1" if paper else "0",
        plot_html=plot_html, spot=spot,
    )

if __name__ == "__main__":
    # Use Agg backend on servers (set via env: MPLBACKEND=Agg)
    app.run(host="127.0.0.1", port=5001, debug=True)

