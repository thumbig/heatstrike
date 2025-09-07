from types import SimpleNamespace
from options_heatmap.snapshots import build_gamma_df

def test_build_gamma_df_handles_dict_and_object():
    details1 = {"strike_price": 155.0, "expiration_date": "2027-06-17"}
    greeks1 = {"gamma": 0.0123}
    snap1 = {"details": details1, "greeks": greeks1}

    details2 = SimpleNamespace(strike_price=160.0, expiration_date="2027-06-24")
    greeks2 = SimpleNamespace(gamma=0.015)
    snap2 = SimpleNamespace(details=details2, greeks=greeks2)

    df = build_gamma_df({"SYM1": snap1, "SYM2": snap2})
    assert {"symbol", "strike", "expiration", "gamma"} <= set(df.columns)
    assert len(df) == 2
    assert set(df["symbol"]) == {"SYM1", "SYM2"}
