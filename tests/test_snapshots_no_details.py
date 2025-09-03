import types
import pandas as pd
from options_heatmap.snapshots import build_gamma_df

def _make_occ(underlying, yymmdd, cp, strike_thou):
    # e.g. "AAPL270617C00155000"
    return f"{underlying}{yymmdd}{cp}{strike_thou:08d}"

def test_build_gamma_df_fallback_when_no_details():
    """
    Snapshot missing 'details' but has greeks.gamma.
    build_gamma_df should parse strike/expiration from OCC symbol.
    """
    sym = _make_occ("AAPL", "270617", "C", 155000)
    # Snapshot-shape: only greeks present, no details
    snap = {
        "latest_quote": {"symbol": sym, "ask_price": 2.91, "bid_price": 2.81},
        "latest_trade": {"symbol": sym, "price": 2.91},
        "greeks": {"gamma": 0.0123},
        # "details": None  # intentionally absent
    }
    df = build_gamma_df({sym: snap})
    assert not df.empty
    row = df.iloc[0]
    assert row["symbol"] == sym
    assert row["strike"] == 155.0
    assert pd.to_datetime(row["expiration"]).year == 2027
    assert abs(row["gamma"] - 0.0123) < 1e-9

def test_build_gamma_df_skips_when_no_greeks():
    """
    Snapshot with latest_quote/trade but no greeks should be skipped.
    """
    sym = _make_occ("AAPL", "270624", "P", 160000)
    snap = {
        "latest_quote": {"symbol": sym, "ask_price": 3.55, "bid_price": 3.45},
        "latest_trade": {"symbol": sym, "price": 3.50},
        # no 'greeks'
    }
    df = build_gamma_df({sym: snap})
    assert df.empty
