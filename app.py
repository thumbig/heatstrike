from __future__ import annotations

import os
import io
import base64
from datetime import datetime
from typing import Optional

import pandas as pd

from flask import (
    Flask, render_template, request, flash, Markup
)

# Headless matplotlib for Seaborn PNG export
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: if seaborn templates aren’t used, still import seaborn plotting funcs
from options_heatmap.core import (
    get_clients, get_spot, fetch_contracts, select_symbols, snapshot,
    prepare_df, nearest_atm_strike
)
from options_heatmap.plot import (
    plot_gamma_heatmap, plot_theta_heatmap, plot_mispriced_heatmap
)

# Plotly for interactive view
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


app = Flask(__name__)
# Use an env secret if present
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")


# ---------------------------
# Helpers
# ---------------------------

def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v in ("1", "true", "True", "on", "yes", "y", "t")

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default

def _to_float(v: Optional[str], default: Optional[float]) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default

def _to_date(v: Optional[str]) -> Optional[pd.Timestamp]:
    if not v:
        return None
    try:
        return pd.to_datetime(v)
    except Exception:
        return None

def _sticky(val, fallback=""):
    return fallback if val is None else val


def _select_and_prepare_df(
    *,
    symbol: str,
    metric: str,
    paper: bool,
    n_expiries: int,
    k_per_expiry: int,
    step: int,
    per_expiry_cap: int,
    strike_min: Optional[float],
    strike_max: Optional[float],
    exp_start: Optional[pd.Timestamp],
    exp_end: Optional[pd.Timestamp],
    strikes_count: Optional[int],
    risk_free_rate: float,
    default_iv: Optional[float],
    share_iv: bool,
    theo_iv_mode: str,
):
    """Build df for any route with identical behavior to CLI."""
    clients = get_clients(paper=paper)
    spot = get_spot(clients.stk_data, symbol)

    cdf = fetch_contracts(
        clients.trade, symbol,
        n_expiries=n_expiries,
        per_expiry_cap=per_expiry_cap
    )

    syms = select_symbols(
        cdf,
        spot=spot,
        n_expiries=n_expiries,
        k_per_expiry=k_per_expiry,
        step=step,
        per_expiry_cap=per_expiry_cap,
        strike_min=strike_min,
        strike_max=strike_max,
        exp_start=exp_start,
        exp_end=exp_end,
        strikes_count=strikes_count,
    )

    snaps = snapshot(clients.opt_data, syms)
    df = prepare_df(
        metric, snaps,
        spot=spot,
        risk_free_rate=risk_free_rate,
        default_iv=default_iv,
        share_iv_across_exp=share_iv,
        theo_iv_mode=theo_iv_mode,
    )
    atm = nearest_atm_strike(df, spot)
    return df, spot, atm


def _make_plotly_heatmaps(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    side_by_side: bool,
    y_descending: bool,
    atm_strike: Optional[float],
    annotate: str,
):
    """
    Build a Plotly heatmap (or two side-by-side for Calls/Puts).
    Hover shows symbol, IV, delta, gamma/theta, theo price as available.
    Cell text: if annotate == 'theo' → theoretical price; 'iv' → IV%; 'symbol' → OCC.
    """
    # Choose value col for colors
    if metric == "mispriced":
        color_col = "mispricing"
        cbar_lbl = "Mispricing ($)"
    elif metric == "theta":
        color_col = "theta"
        cbar_lbl = "Theta"
    else:
        color_col = "gamma"
        cbar_lbl = "Gamma"

    # Helper: pivot utility
    def _pivot(df_sub: pd.DataFrame, col: str) -> pd.DataFrame:
        if df_sub is None or df_sub.empty or col not in df_sub.columns:
            return pd.DataFrame()
        pv = df_sub.pivot_table(index="strike", columns="expiration", values=col, aggfunc="mean")
        # order expiries
        if len(pv.columns) > 0:
            pv = pv.reindex(sorted(pv.columns), axis=1)
        # order strikes
        strikes = sorted(set([float(x) for x in pv.index]))
        if y_descending:
            strikes = list(reversed(strikes))
        return pv.reindex(strikes)

    # Annotation matrix (string)
    def _annot(df_sub: pd.DataFrame, base: pd.DataFrame) -> Optional[np.ndarray]:
        if annotate == "theo" and "theo_price" in df_sub.columns:
            pv = _pivot(df_sub, "theo_price")
            pv = pv.reindex(index=base.index, columns=base.columns)
            a = np.full(pv.shape, "", dtype=object)
            it = np.nditer(pv.to_numpy(dtype=float), flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                v = it[0]
                if np.isfinite(v):
                    a[idx] = f"{float(v):.2f}"
                it.iternext()
            return a
        elif annotate == "iv" and "iv" in df_sub.columns:
            pv = _pivot(df_sub, "iv")
            pv = pv.reindex(index=base.index, columns=base.columns)
            a = np.full(pv.shape, "", dtype=object)
            it = np.nditer(pv.to_numpy(dtype=float), flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                v = it[0]
                if np.isfinite(v):
                    a[idx] = f"{float(v)*100:.1f}"
                it.iternext()
            return a
        elif annotate == "symbol":
            tmp = df_sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
            pv = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            pv = pv.reindex(index=base.index, columns=base.columns, fill_value="")
            return pv.to_numpy(dtype=object)
        else:
            return None

    # Common layout
    def _layout(fig, title):
        fig.update_layout(
            title=title,
            margin=dict(l=60, r=30, t=60, b=60),
            xaxis_title="Expiration (MM/DD)",
            yaxis_title="Strike",
        )

    # Build one or two panes
    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_c = _pivot(calls, color_col)
        pv_p = _pivot(puts,  color_col)

        # Ensure we always have at least an empty grid to render
        left_rows = pv_c.index if not pv_c.empty else pv_p.index
        right_rows = pv_p.index if not pv_p.empty else pv_c.index

        # Align rows for shared y
        if pv_c.empty:
            pv_c = pd.DataFrame(index=left_rows)
        if pv_p.empty:
            pv_p = pd.DataFrame(index=right_rows)
        # final align
        all_rows = sorted(set(pv_c.index.tolist()) | set(pv_p.index.tolist()))
        if y_descending:
            all_rows = list(reversed(all_rows))
        pv_c = pv_c.reindex(all_rows)
        pv_p = pv_p.reindex(all_rows)

        # Build annotations
        ann_c = _annot(calls, pv_c)
        ann_p = _annot(puts,  pv_p)

        # X tick labels: mm/dd
        def _mmdd(cols): return [pd.to_datetime(c).strftime("%m/%d") for c in cols] if len(cols) else []

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                            subplot_titles=("Calls", "Puts"))

        if pv_c.shape[0] and pv_c.shape[1]:
            zc = pv_c.to_numpy(dtype=float)
            text_c = ann_c if ann_c is not None else None
            fig.add_trace(go.Heatmap(
                z=zc, x=_mmdd(pv_c.columns), y=pv_c.index.astype(str),
                colorscale="RdBu" if metric == "theta" else "RdBu" if metric == "mispriced" else "RdBu",
                zmid=0, text=text_c, texttemplate="%{text}", hovertemplate="%{y} / %{x}<extra></extra>"
            ), row=1, col=1)

        if pv_p.shape[0] and pv_p.shape[1]:
            zp = pv_p.to_numpy(dtype=float)
            text_p = ann_p if ann_p is not None else None
            fig.add_trace(go.Heatmap(
                z=zp, x=_mmdd(pv_p.columns), y=pv_p.index.astype(str),
                colorscale="RdBu" if metric == "theta" else "RdBu" if metric == "mispriced" else "RdBu",
                zmid=0, text=text_p, texttemplate="%{text}", hovertemplate="%{y} / %{x}<extra></extra>"
            ), row=1, col=2)

        # ATM line as a shape (if available)
        if atm_strike is not None and len(all_rows):
            # y coordinate is the strike label string; choose nearest index
            diffs = np.abs(np.array(all_rows, dtype=float) - float(atm_strike))
            y_label = str(all_rows[int(diffs.argmin())])
            for c in (1, 2):
                fig.add_hline(y=y_label, line_dash="dash", line_color="black", line_width=2, row=1, col=c)

        _layout(fig, title)
        # more informative hover: we’ll add hovertext via customdata if needed later
        return fig

    # Single-pane
    pv = _pivot(df, color_col)
    ann = _annot(df, pv)

    fig = go.Figure()
    if pv.shape[0] and pv.shape[1]:
        z = pv.to_numpy(dtype=float)
        text = ann if ann is not None else None
        fig.add_trace(go.Heatmap(
            z=z, x=[pd.to_datetime(c).strftime("%m/%d") for c in pv.columns], y=pv.index.astype(str),
            colorscale="RdBu" if metric == "theta" else "RdBu" if metric == "mispriced" else "RdBu",
            zmid=0, text=text, texttemplate="%{text}", hovertemplate="%{y} / %{x}<extra></extra>"
        ))
    if atm_strike is not None and pv.shape[0]:
        diffs = np.abs(pv.index.to_numpy(dtype=float) - float(atm_strike))
        y_label = str(pv.index[int(diffs.argmin())])
        fig.add_hline(y=y_label, line_dash="dash", line_color="black", line_width=2)

    _layout(fig, title)
    return fig


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def index():
    return Markup(
        '<h2>Heatstrike</h2>'
        '<p><a href="/seaborn">Seaborn View</a> &nbsp; | &nbsp; '
        '<a href="/plotly">Plotly View</a></p>'
    )


@app.get("/seaborn")
def seaborn_view():
    # Query params + defaults
    symbol = (request.args.get("symbol") or os.environ.get("DEFAULT_SYMBOL", "AAPL")).upper()
    metric = (request.args.get("metric") or os.environ.get("DEFAULT_METRIC", "mispriced")).lower()
    paper = _to_bool(request.args.get("paper"), default=False)
    side_by_side = request.args.get("side_by_side", "1") == "1"
    y_order = request.args.get("y_order", "desc")
    annotate = request.args.get("annotate", "iv")
    n_expiries = _to_int(request.args.get("n_expiries"), int(os.environ.get("N_EXPIRIES", 9)))
    k_per_expiry = _to_int(request.args.get("k_per_expiry"), int(os.environ.get("K_PER_EXPIRY", 11)))
    step = _to_int(request.args.get("step"), int(os.environ.get("EVERY_OTHER_STEP", 2)))
    per_expiry_cap = _to_int(request.args.get("per_expiry_cap"), int(os.environ.get("PER_EXPIRY_CAP", 10)))
    save = request.args.get("save", "")

    # Constraints
    strike_min = _to_float(request.args.get("strike_min"), None)
    strike_max = _to_float(request.args.get("strike_max"), None)
    exp_start = _to_date(request.args.get("exp_start"))
    exp_end = _to_date(request.args.get("exp_end"))
    strikes_count = _to_int(request.args.get("strikes_count"), None) if request.args.get("strikes_count") else None

    try:
        df, spot, atm = _select_and_prepare_df(
            symbol=symbol, metric=metric, paper=paper,
            n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
            strike_min=strike_min, strike_max=strike_max,
            exp_start=exp_start, exp_end=exp_end, strikes_count=strikes_count,
            risk_free_rate=float(os.environ.get("RISK_FREE_RATE", "0.02")),
            default_iv=(float(os.environ["DEFAULT_IV"]) if "DEFAULT_IV" in os.environ else None),
            share_iv=os.environ.get("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False"),
            theo_iv_mode=os.environ.get("THEO_IV_MODE", "per_expiry_mean"),
        )

        # Build Seaborn PNG via our plotters
        title = f"{metric.title()} — {symbol} (Spot ${spot:.2f})"
        y_desc = (y_order == "desc")

        if metric == "mispriced":
            fig_or_ax = plot_mispriced_heatmap(
                df, title=title, save_path=(save or None),
                side_by_side=side_by_side, annotate=("theo" if annotate == "iv" else annotate),
                annotate_fmt=".2f", use_pct=False, atm_strike=atm, y_descending=y_desc
            )
        elif metric == "gamma":
            fig_or_ax = plot_gamma_heatmap(
                df, title=title, save_path=(save or None),
                side_by_side=side_by_side, annotate=("iv" if annotate not in ("symbol","none") else annotate),
                annotate_fmt=".1f", atm_strike=atm, y_descending=y_desc
            )
        else:
            fig_or_ax = plot_theta_heatmap(
                df, title=title, save_path=(save or None),
                side_by_side=side_by_side, annotate=("iv" if annotate not in ("symbol","none") else annotate),
                annotate_fmt=".1f", atm_strike=atm, y_descending=y_desc
            )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")

        return render_template(
            "seaborn.html",
            figure_png=b64,
            symbol=symbol, metric=metric, paper=paper,
            side_by_side=side_by_side, y_order=y_order, annotate=annotate,
            n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
            save=save, spot=spot,
            strike_min=_sticky(strike_min, ""), strike_max=_sticky(strike_max, ""),
            exp_start_val=(exp_start.date().isoformat() if exp_start is not None else ""),
            exp_end_val=(exp_end.date().isoformat() if exp_end is not None else ""),
            strikes_count=_sticky(strikes_count, ""),
        )
    except Exception as e:
        return render_template(
            "seaborn.html",
            figure_png=None,
            symbol=symbol, metric=metric, paper=paper,
            side_by_side=side_by_side, y_order=y_order, annotate=annotate,
            n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
            save=save, error=str(e),
            strike_min=_sticky(strike_min, ""), strike_max=_sticky(strike_max, ""),
            exp_start_val=(request.args.get("exp_start") or ""),
            exp_end_val=(request.args.get("exp_end") or ""),
            strikes_count=_sticky(strikes_count, ""),
        )


@app.get("/plotly")
def plotly_view():
    symbol = (request.args.get("symbol") or os.environ.get("DEFAULT_SYMBOL", "AAPL")).upper()
    metric = (request.args.get("metric") or os.environ.get("DEFAULT_METRIC", "mispriced")).lower()
    paper = _to_bool(request.args.get("paper"), default=False)
    side_by_side = request.args.get("side_by_side", "1") == "1"
    y_order = request.args.get("y_order", "desc")
    annotate = request.args.get("annotate", "iv")  # 'iv'|'theo'|'symbol'|'none'

    # selection params
    n_expiries = _to_int(request.args.get("n_expiries"), int(os.environ.get("N_EXPIRIES", 9)))
    k_per_expiry = _to_int(request.args.get("k_per_expiry"), int(os.environ.get("K_PER_EXPIRY", 11)))
    step = _to_int(request.args.get("step"), int(os.environ.get("EVERY_OTHER_STEP", 2)))
    per_expiry_cap = _to_int(request.args.get("per_expiry_cap"), int(os.environ.get("PER_EXPIRY_CAP", 10)))

    # constraints
    strike_min = _to_float(request.args.get("strike_min"), None)
    strike_max = _to_float(request.args.get("strike_max"), None)
    exp_start = _to_date(request.args.get("exp_start"))
    exp_end = _to_date(request.args.get("exp_end"))
    strikes_count = _to_int(request.args.get("strikes_count"), None) if request.args.get("strikes_count") else None

    try:
        df, spot, atm = _select_and_prepare_df(
            symbol=symbol, metric=metric, paper=paper,
            n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
            strike_min=strike_min, strike_max=strike_max,
            exp_start=exp_start, exp_end=exp_end, strikes_count=strikes_count,
            risk_free_rate=float(os.environ.get("RISK_FREE_RATE", "0.02")),
            default_iv=(float(os.environ["DEFAULT_IV"]) if "DEFAULT_IV" in os.environ else None),
            share_iv=os.environ.get("SHARE_IV_ACROSS_EXP", "1") not in ("0", "false", "False"),
            theo_iv_mode=os.environ.get("THEO_IV_MODE", "per_expiry_mean"),
        )

        title = f"{metric.title()} — {symbol} (Spot ${spot:.2f})"
        fig = _make_plotly_heatmaps(
            df,
            metric=metric,
            title=title,
            side_by_side=side_by_side,
            y_descending=(y_order == "desc"),
            atm_strike=atm,
            annotate=annotate,
        )
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # If you have a template, render it; else send simple HTML
        try:
            return render_template(
                "plotly.html",
                plot_html=Markup(plot_html),
                symbol=symbol, metric=metric, paper=paper,
                side_by_side=side_by_side, y_order=y_order, annotate=annotate,
                n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
                strike_min=_sticky(strike_min, ""), strike_max=_sticky(strike_max, ""),
                exp_start_val=(exp_start.date().isoformat() if exp_start is not None else ""),
                exp_end_val=(exp_end.date().isoformat() if exp_end is not None else ""),
                strikes_count=_sticky(strikes_count, ""),
                spot=spot,
            )
        except Exception:
            # minimal fallback if template missing
            return (
                "<html><head><meta charset='utf-8'><title>Plotly Matrix</title></head>"
                "<body>"
                f"<h3>{title}</h3>"
                f"{plot_html}"
                "</body></html>"
            )

    except Exception as e:
        return render_template(
            "plotly.html",
            plot_html=Markup(f"<div style='color:#b00020;font-weight:600;'>{str(e)}</div>"),
            symbol=symbol, metric=metric, paper=paper,
            side_by_side=side_by_side, y_order=y_order, annotate=annotate,
            n_expiries=n_expiries, k_per_expiry=k_per_expiry, step=step, per_expiry_cap=per_expiry_cap,
            strike_min=_sticky(strike_min, ""), strike_max=_sticky(strike_max, ""),
            exp_start_val=(request.args.get("exp_start") or ""),
            exp_end_val=(request.args.get("exp_end") or ""),
            strikes_count=_sticky(strikes_count, ""),
            spot=None,
        )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)


# ---------------------------
# Test-friendly main() (used by pytest)
# ---------------------------

def main(*, trade=None, opt_data=None, stk_data=None, paper=True):
    """
    Build the DataFrame once without starting Flask.
    Accepts injected dummy clients; falls back to ENV-based clients.
    Preserves test expectation: when metric=='gamma' and no greeks,
    raise 'No gamma available'.
    """
    from options_heatmap.core import (
        get_clients, get_spot, fetch_contracts, select_symbols, snapshot, prepare_df
    )

    symbol = os.environ.get("DEFAULT_SYMBOL", "AAPL")
    metric = os.environ.get("DEFAULT_METRIC", "gamma").lower()
    n_expiries = int(os.environ.get("N_EXPIRIES", "9"))
    k_per_expiry = int(os.environ.get("K_PER_EXPIRY", "11"))
    step = int(os.environ.get("EVERY_OTHER_STEP", "2"))
    per_expiry_cap = int(os.environ.get("PER_EXPIRY_CAP", "10"))
    rfr = float(os.environ.get("RISK_FREE_RATE", "0.02"))
    default_iv = os.environ.get("DEFAULT_IV")
    default_iv_f = float(default_iv) if default_iv not in (None, "") else None
    share_iv = os.environ.get("SHARE_IV_ACROSS_EXP", "1") not in ("0","false","False")
    theo_iv_mode = os.environ.get("THEO_IV_MODE", "per_expiry_mean")

    # Constraints via env (optional)
    strike_min = os.environ.get("STRIKE_MIN")
    strike_max = os.environ.get("STRIKE_MAX")
    exp_start = os.environ.get("EXP_START")
    exp_end = os.environ.get("EXP_END")
    strikes_count = os.environ.get("STRIKES_COUNT")
    strike_min_f = float(strike_min) if strike_min not in (None, "") else None
    strike_max_f = float(strike_max) if strike_max not in (None, "") else None
    exp_start_v = pd.to_datetime(exp_start) if exp_start else None
    exp_end_v = pd.to_datetime(exp_end) if exp_end else None
    strikes_count_i = int(strikes_count) if strikes_count not in (None, "") else None

    # Build clients if not injected
    if trade is None or opt_data is None or stk_data is None:
        clients = get_clients(paper=paper)
        trade, opt_data, stk_data = clients.trade, clients.opt_data, clients.stk_data

    spot = get_spot(stk_data, symbol)
    cdf = fetch_contracts(trade, symbol, n_expiries=n_expiries, per_expiry_cap=per_expiry_cap)
    syms = select_symbols(
        cdf, spot=spot, n_expiries=n_expiries, k_per_expiry=k_per_expiry,
        step=step, per_expiry_cap=per_expiry_cap,
        strike_min=strike_min_f, strike_max=strike_max_f,
        exp_start=exp_start_v, exp_end=exp_end_v, strikes_count=strikes_count_i,
    )
    snaps = snapshot(opt_data, syms)
    try:
        df = prepare_df(
            metric, snaps, spot=spot, risk_free_rate=rfr,
            default_iv=default_iv_f, share_iv_across_exp=share_iv, theo_iv_mode=theo_iv_mode
        )
        return df
    except RuntimeError as err:
        msg = str(err)
        if metric == "gamma" and "No greeks available" in msg:
            raise RuntimeError("No gamma available") from err
        raise
