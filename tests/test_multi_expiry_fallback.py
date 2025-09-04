import pandas as pd
from options_heatmap.snapshots import build_gamma_df
from options_heatmap.plot import plot_gamma_heatmap

def _sym(und, yymmdd, cp, strike_thou):
    return f"{und}{yymmdd}{cp}{strike_thou:08d}"

def test_multiple_expiries_via_default_iv(tmp_path):
    """
    Only one expiry has IV; the other lacks both greeks.gamma and IV.
    With default_iv set, we still compute gamma for both expiries.
    """
    # 2027-06-17 has IV (call), 2027-06-24 lacks IV (put)
    s1 = _sym("AAPL", "270617", "C", 150000)
    s2 = _sym("AAPL", "270624", "P", 150000)
    snaps = {
        s1: {"implied_volatility": 0.25},  # has IV → compute gamma
        s2: {},                             # no greeks, no IV → use default_iv
    }
    df = build_gamma_df(
        snaps,
        spot=155.0,
        risk_free_rate=0.02,
        now_ts=pd.Timestamp("2025-01-01", tz="UTC"),
        default_iv=0.22,
        share_iv_across_exp=True,
    )
    # Both expiries present
    assert set(pd.to_datetime(df["expiration"]).dt.date) >= {
        pd.Timestamp("2027-06-17").date(), pd.Timestamp("2027-06-24").date()
    }
    out = tmp_path / "multi.png"
    fig = plot_gamma_heatmap(df, title="Gamma", save_path=str(out), side_by_side=True)
    assert fig is not None and out.exists()
