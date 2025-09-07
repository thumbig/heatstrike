from __future__ import annotations
import os
from flask import Flask
from markupsafe import Markup

from web.routes import ui

from options_heatmap.core import (
    get_clients, get_spot, fetch_contracts, select_symbols, snapshot, prepare_df
)
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")
app.register_blueprint(ui)

@app.get("/")
def index():
    return Markup(
        '<h2>Heatstrike</h2>'
        '<p><a href="/seaborn">Seaborn View</a> &nbsp; | &nbsp; '
        '<a href="/plotly">Plotly View</a></p>'
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

# -------- Test-friendly entrypoint --------
def main(*, trade=None, opt_data=None, stk_data=None, paper=True):
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
        if metric == "gamma" and "No greeks available" in str(err):
            raise RuntimeError("No gamma available") from err
        raise


