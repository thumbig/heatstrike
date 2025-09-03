# Only the parts that compute ATM and pass it into the plotting functions change.
# Paste this whole file if easier; otherwise, the two helper functions and calls are the key changes.

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request, send_from_directory, flash, session

# Alpaca + your modules (gvars/pvars based)
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

UNDERLYING_DEFAULT = "AAPL"
OUTPUT_DIR = Path("static")
OUTPUT_DIR.mkdir(exist_ok=True)

N_EXPIRIES = int(os.getenv("N_EXPIRIES", "9"))
K_PER_EXPIRY = int(os.getenv("K_PER_EXPIRY", "11"))
EVERY_OTHER_STEP = int(os.getenv("EVERY_OTHER_STEP", "2"))
PER_EXPIRY_CAP = int(os.getenv("PER_EXPIRY_CAP", "10"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02"))

DEFAULT_IV = os.getenv("DEFAULT_IV")
DEFAULT_IV_F = float(DEFAULT_IV) if DEFAULT_IV not in (None, "") else None
SHARE_IV_ACROSS_EXP = os.getenv("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False")
THEO_IV_MODE = os.getenv("THEO_IV_MODE", "per_expiry_mean")

ALLOWED_UNDERLYINGS = ["SPY", "AAPL", "TSLA", "NVDA", "QQQ", "IWM"]  # (skip SPX/SPXW)

app = Flask(__name__)
app.secret_key = "dev-secret"  # replace as desired

def _get_clients(paper: bool = True):
    if paper:
        import gvars as cred
        API_KEY = cred.ALPACA_API_KEY_2
        API_SECRET = cred.ALPACA_SECRET_KEY_2
    else:
        import pvars as cred
        API_KEY = cred.ALPACA_API_KEY
        API_SECRET = cred.ALPACA_SECRET_KEY
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

def _nearest_atm_strike(df: pd.DataFrame, spot: float) -> Optional[float]:
    """Pick the plotted strike closest to spot."""
    try:
        strikes = pd.Series(sorted(df["strike"].unique()))
        return float(strikes.iloc[(strikes - spot).abs().argmin()])
    except Exception:
        return None

def _prepare_df(metric: str, symbol: str, paper: bool):
    trade, opt_data, stk_data = _get_clients(paper=paper)
    spot = _get_spot(stk_data, symbol)
    cdf = _balanced_contracts(trade, symbol)
    syms = _select_symbols(cdf, spot)
    snaps = _snapshot(opt_data, syms)
    if metric == "mispriced":
        df = build_pricing_df(
            snaps, spot=spot, risk_free_rate=RISK_FREE_RATE,
            default_iv=DEFAULT_IV_F, share_iv_across_exp=SHARE_IV_ACROSS_EXP,
            theo_iv_mode=THEO_IV_MODE,
        )
        if df.empty:
            raise RuntimeError("No prices/IVs available to compute mispricing.")
    else:
        df = build_gamma_df(
            snaps, spot=spot, risk_free_rate=RISK_FREE_RATE,
            default_iv=DEFAULT_IV_F, share_iv_across_exp=SHARE_IV_ACROSS_EXP,
        )
        if df.empty:
            raise RuntimeError("No greeks available (even after fallback).")
    return df, spot

def _render_and_save(metric: str, underlying: str, paper: bool = True) -> Optional[str]:
    df, spot = _prepare_df(metric, underlying, paper)
    atm = _nearest_atm_strike(df, spot)
    stamp = int(time.time())
    fname = f"heatmap_{metric}_{underlying}_{stamp}.png"
    fpath = OUTPUT_DIR / fname
    if metric == "mispriced":
        fig = plot_mispriced_heatmap(
            df, title=f"Mispriced — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="theo", annotate_fmt=".2f", use_pct=False,
            atm_strike=atm,
        )
    elif metric == "gamma":
        fig = plot_gamma_heatmap(
            df, title=f"Gamma — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="iv", annotate_fmt=".1f",
            atm_strike=atm,
        )
    else:
        fig = plot_theta_heatmap(
            df, title=f"Theta — {underlying}", save_path=str(fpath),
            side_by_side=True, annotate="iv", annotate_fmt=".1f",
            atm_strike=atm,
        )
    if fig:
        return f"static/{fname}", spot, atm
    return None, spot, atm

def _load_from_session():
    symbol = request.values.get("symbol") or session.get("symbol") or UNDERLYING_DEFAULT
    metric = (request.values.get("metric") or session.get("metric") or "mispriced").lower()
    paper = request.values.get("paper")
    if paper is None:
        paper = session.get("paper", "1")
    return symbol.upper(), metric, (paper in ("1","true","True")), paper

@app.route("/", methods=["GET", "POST"])
def index():
    symbol, metric, paper_bool, paper_raw = _load_from_session()
    dropdown = request.values.get("dropdown_symbol", "")
    typed = request.values.get("symbol", "").strip()
    if not typed and dropdown:
        symbol = dropdown.upper()

    if request.method == "POST":
        session["symbol"] = symbol
        session["metric"] = metric
        session["paper"] = "1" if paper_bool else "0"
        try:
            img_rel, spot, atm = _render_and_save(metric, symbol, paper=paper_bool)
            if not img_rel:
                flash("No figure produced.", "warning")
            return render_template(
                "index.html",
                allowed=ALLOWED_UNDERLYINGS,
                selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
                symbol=symbol, metric=metric, image_path=img_rel,
                download_href=img_rel, paper=("1" if paper_bool else "0"),
                spot=spot, atm=atm,
            )
        except Exception as e:
            flash(str(e), "error")

    # GET: show remembered selections + spot if available
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
        paper=("1" if paper_bool else "0"), spot=spot_val, atm=None,
    )

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ---------------------- Interactive (Plotly) ----------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def _format_mmdd(seq):
    return [pd.to_datetime(c).strftime("%m/%d") for c in seq]

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    pv = df.pivot_table(index="strike", columns="expiration", values=value_col, aggfunc="mean")
    pv = pv.sort_index(axis=0, ascending=False)  # strikes descending
    if len(pv.columns) > 0:
        pv = pv.reindex(sorted(pv.columns), axis=1)
    return pv

def _sig3(x):
    try:
        if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))):
            return ""
        from math import log10, floor
        if x == 0: return "0"
        mag = floor(log10(abs(x)))
        places = max(0, 2 - mag)
        return f"{x:.{places}f}"
    except Exception:
        return ""

def _hover_text_matrix(df: pd.DataFrame, value_col: str, extra_cols: list) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = ["strike", "expiration", "symbol", value_col] + [c for c in extra_cols if c in df.columns]
    tmp = df[cols].copy()
    tmp["expiration"] = pd.to_datetime(tmp["expiration"])
    def row_text(r):
        bits = [f"<b>{r['symbol']}</b>",
                f"Strike: {_sig3(r['strike'])}",
                f"Exp: {pd.to_datetime(r['expiration']).strftime('%Y-%m-%d')}"]
        if value_col in r and pd.notna(r[value_col]):
            bits.append(f"{value_col.title()}: {_sig3(r[value_col])}")
        for k in extra_cols:
            if k in r and pd.notna(r[k]):
                if k in ("iv","iv_market","iv_theo"):
                    bits.append(f"{k.replace('_',' ').title()}: {_sig3(100*r[k])}%")
                elif k in ("theo_price","market_price","mispricing"):
                    bits.append(f"{k.replace('_',' ').title()}: ${_sig3(r[k])}")
                else:
                    bits.append(f"{k.replace('_',' ').title()}: {_sig3(r[k])}")
        return "<br>".join(bits)
    tmp["hover"] = tmp.apply(row_text, axis=1)
    hov = tmp.pivot_table(index="strike", columns="expiration", values="hover", aggfunc="first")
    hov = hov.sort_index(axis=0, ascending=False)
    if len(hov.columns) > 0:
        hov = hov.reindex(sorted(hov.columns), axis=1)
    return hov

@app.route("/interactive", methods=["GET", "POST"])
def interactive():
    symbol = request.values.get("symbol") or session.get("symbol") or UNDERLYING_DEFAULT
    metric = (request.values.get("metric") or session.get("metric") or "mispriced").lower()
    paper_bool = (request.values.get("paper") or session.get("paper","1")) in ("1","true","True")
    dropdown = request.values.get("dropdown_symbol", "")
    typed = request.values.get("symbol", "").strip()
    if not typed and dropdown:
        symbol = dropdown.upper()

    session["symbol"] = symbol
    session["metric"] = metric
    session["paper"] = "1" if paper_bool else "0"

    try:
        df, spot = _prepare_df(metric, symbol, paper=paper_bool)
    except Exception as e:
        flash(str(e), "error")
        return render_template("interactive.html",
            allowed=ALLOWED_UNDERLYINGS, selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
            symbol=symbol, metric=metric, paper=("1" if paper_bool else "0"),
            plot_html="<p>Error generating plot.</p>", spot=None)

    value_col = {"mispriced":"mispricing", "gamma":"gamma", "theta":"theta"}[metric]
    has_type = "type" in df.columns
    calls = df[df["type"] == "call"] if has_type else df
    puts  = df[df["type"] == "put"]  if has_type else pd.DataFrame(columns=df.columns)

    pv_c = _pivot(calls, value_col)
    pv_p = _pivot(puts,  value_col)

    x_c = _format_mmdd(pv_c.columns) if not pv_c.empty else []
    x_p = _format_mmdd(pv_p.columns) if not pv_p.empty else []
    y_c = list(pv_c.index) if not pv_c.empty else []
    y_p = list(pv_p.index) if not pv_p.empty else y_c

    # ATM strike for overlay line
    def nearest_atm(strikes: list, s: float) -> Optional[float]:
        if not strikes: return None
        arr = np.array(strikes, dtype=float)
        return float(arr[np.abs(arr - s).argmin()])
    atm = nearest_atm(sorted(df["strike"].unique()), spot)

    extra_cols = []
    if metric in ("gamma", "theta"):
        extra_cols = ["iv","delta", metric]
    elif metric == "mispriced":
        extra_cols = ["theo_price","market_price","iv_theo","iv_market","mispricing_pct","iv","delta","gamma","theta"]

    hov_c = _hover_text_matrix(calls, value_col, extra_cols) if not calls.empty else pd.DataFrame()
    hov_p = _hover_text_matrix(puts,  value_col, extra_cols) if not puts.empty else pd.DataFrame()
    hover_c = hov_c.values if not hov_c.empty else None
    hover_p = hov_p.values if not hov_p.empty else None

    text_c = text_p = None
    texttemplate = None
    if metric == "mispriced":
        def masked_text(pv):
            if pv.empty: return None
            arr = pv.values.astype(float)
            mask = ~np.isfinite(arr)
            arr = arr.copy(); arr[mask] = np.nan
            return arr
        theo_c = _pivot(calls, "theo_price"); text_c = masked_text(theo_c)
        theo_p = _pivot(puts,  "theo_price"); text_p = masked_text(theo_p)
        texttemplate = "$%{text:.2f}"

    cmap = "Bluered" if metric in ("mispriced","gamma") else "RdBu"
    zmid = 0

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if has_type:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Calls", "Puts"))
        if not pv_c.empty:
            fig.add_trace(go.Heatmap(z=pv_c.values, x=x_c, y=y_c, colorscale=cmap, zmid=zmid,
                                     colorbar=dict(title=value_col.title()),
                                     hoverinfo="text", hovertext=hover_c,
                                     text=text_c, texttemplate=texttemplate), row=1, col=1)
            # ATM line as a Scatter over categories
            if atm is not None and x_c:
                fig.add_trace(go.Scatter(x=[x_c[0], x_c[-1]], y=[atm, atm],
                                         mode="lines", line=dict(color="black", width=2, dash="dash"),
                                         hoverinfo="skip", showlegend=False), row=1, col=1)
        if not pv_p.empty:
            fig.add_trace(go.Heatmap(z=pv_p.values, x=x_p, y=y_p, colorscale=cmap, zmid=zmid,
                                     showscale=False, hoverinfo="text", hovertext=hover_p,
                                     text=text_p, texttemplate=texttemplate), row=1, col=2)
            if atm is not None and x_p:
                fig.add_trace(go.Scatter(x=[x_p[0], x_p[-1]], y=[atm, atm],
                                         mode="lines", line=dict(color="black", width=2, dash="dash"),
                                         hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text="Strike", row=1, col=1, autorange="reversed")
        fig.update_xaxes(title_text="Expiration (MM/DD)", row=1, col=1)
        fig.update_xaxes(title_text="Expiration (MM/DD)", row=1, col=2)
        fig.update_layout(title=f"{value_col.title()} — {symbol}  (Spot ${spot:.2f})",
                          margin=dict(l=70, r=20, t=60, b=60), height=640)
    else:
        fig = go.Figure(
            data=go.Heatmap(z=pv_c.values, x=x_c, y=y_c, colorscale=cmap, zmid=zmid,
                            colorbar=dict(title=value_col.title()),
                            hoverinfo="text", hovertext=hover_c,
                            text=text_c, texttemplate=texttemplate)
        )
        if atm is not None and x_c:
            fig.add_trace(go.Scatter(x=[x_c[0], x_c[-1]], y=[atm, atm],
                                     mode="lines", line=dict(color="black", width=2, dash="dash"),
                                     hoverinfo="skip", showlegend=False))
        fig.update_yaxes(title_text="Strike", autorange="reversed")
        fig.update_xaxes(title_text="Expiration (MM/DD)")
        fig.update_layout(title=f"{value_col.title()} — {symbol}  (Spot ${spot:.2f})",
                          margin=dict(l=70, r=20, t=60, b=60), height=640)

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return render_template(
        "interactive.html",
        allowed=ALLOWED_UNDERLYINGS,
        selected_dropdown=(symbol if symbol in ALLOWED_UNDERLYINGS else ""),
        symbol=symbol, metric=metric, paper=("1" if paper_bool else "0"),
        plot_html=plot_html, spot=spot,
    )

if __name__ == "__main__":
    # Use Agg backend on servers (set via env: MPLBACKEND=Agg)
    app.run(host="127.0.0.1", port=5001, debug=True)

