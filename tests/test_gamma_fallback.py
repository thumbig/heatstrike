import types
import pandas as pd
from options_heatmap.snapshots import build_gamma_df
from options_heatmap.plot import plot_gamma_heatmap

def _make_occ(und, yymmdd, cp, strike_thou):
    return f"{und}{yymmdd}{cp}{strike_thou:08d}"

def test_fallback_gamma_and_side_by_side_plot(tmp_path):
    # Two symbols, same strikes but different types; only IV provided (no greeks.gamma)
    sym_call = _make_occ("AAPL", "270617", "C", 150000)
    sym_put  = _make_occ("AAPL", "270617", "P", 150000)
    snap_call = {"implied_volatility": 0.25}  # no details, no greeks → BS fallback
    snap_put  = {"implied_volatility": 0.25}
    snaps = {sym_call: snap_call, sym_put: snap_put}

    df = build_gamma_df(snaps, spot=155.0, risk_free_rate=0.02, now_ts=pd.Timestamp("2025-01-01"))
    assert not df.empty
    # Both sides present
    assert set(df["type"]) == {"call", "put"}
    # Whole-number strike present
    assert "strike" in df.columns and df["strike"].iloc[0] == 150

    # Side-by-side figure that saves to file
    out = tmp_path / "gamma.png"
    fig = plot_gamma_heatmap(df, title="Gamma", save_path=str(out), side_by_side=True)
    assert fig is not None
    assert out.exists()

