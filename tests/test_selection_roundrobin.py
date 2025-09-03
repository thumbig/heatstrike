import pandas as pd
from collections import Counter
from options_heatmap.selection import select_symbols_balanced

def _sym(exp, strike, t):
    # embed expiry as YYYYMMDD to avoid '-' parsing issues
    exp_id = pd.Timestamp(exp).strftime("%Y%m%d")
    return f"{t}-{exp_id}-{strike}"

def test_select_symbols_balanced_roundrobin_distribution():
    # Build a toy chain: 3 expiries, 20 calls + 20 puts each, strikes 90..109
    rows = []
    expiries = [pd.Timestamp("2027-06-17"), pd.Timestamp("2027-06-24"), pd.Timestamp("2027-07-01")]
    for exp in expiries:
        for t in ["call", "put"]:
            for k in range(20):
                rows.append({
                    "symbol": _sym(exp, 90 + k, t[0]),
                    "type": t,
                    "strike": 90 + k,
                    "expiration": exp,
                })
    cdf = pd.DataFrame(rows)
    sel = select_symbols_balanced(
        cdf, spot=100, n_expiries=3, k_per_expiry=10, step=2, cap=12, per_expiry_cap=4
    )
    # group by the middle token YYYYMMDD
    exp_counts = Counter(s.split("-")[1] for s in sel)
    assert all(v <= 4 for v in exp_counts.values())
    assert len(sel) == 12

