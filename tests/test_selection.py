import pandas as pd
from options_heatmap.selection import nearest_indices_every_other, select_symbols_both_sides

def test_nearest_indices_every_other():
    idxs = nearest_indices_every_other([90, 95, 100, 105, 110], center_value=101, k=3, step=2)
    assert len(idxs) == 3
    assert 2 in idxs  # index of 100

def test_select_symbols_both_sides_respects_cap():
    rows = []
    for t in ["call", "put"]:
        for k in range(10):
            rows.append({"symbol": f"SYM{t[0]}{k}", "type": t, "strike": 95 + 5*k, "expiration": pd.Timestamp("2027-06-17")})
    cdf = pd.DataFrame(rows)
    syms = select_symbols_both_sides(cdf, spot=120, n_expiries=1, k_per_expiry=10, step=2, cap=12)
    assert len(syms) <= 12
    assert any(s.startswith("SYMc") or s.startswith("SYMp") or "SYM" in s for s in syms)
