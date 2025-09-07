import pytest
import pandas as pd

import app as app_mod

class DummyTradeClient:
    def __init__(self): pass

class DummyStockHistoricalDataClient:
    def __init__(self, price): self.price = price
    def get_stock_latest_trade(self, req): return type("LT", (), {"price": self.price})

class DummyOptionHistoricalDataClient:
    def __init__(self): pass
    def get_option_snapshot(self, req): return {}  # no snaps -> no greeks

def test_app_main_maps_greeks_error_to_gamma(monkeypatch):
    # Avoid default IV rescuing the path
    monkeypatch.delenv("DEFAULT_IV", raising=False)
    # Force metric
    monkeypatch.setenv("DEFAULT_METRIC", "gamma")

    trade = DummyTradeClient()
    stk = DummyStockHistoricalDataClient(price=150.0)
    opt = DummyOptionHistoricalDataClient()

    # Stub called functions to avoid Alpaca
    monkeypatch.setattr(app_mod, "get_clients", lambda paper=True: None)
    monkeypatch.setattr(app_mod, "get_spot", lambda s, sym: 150.0)
    # minimal contracts df
    import pandas as pd
    cdf = pd.DataFrame({"symbol":["X1","X2"], "strike":[150,155], "expiration":pd.to_datetime(["2027-06-17","2027-06-24"])})
    monkeypatch.setattr(app_mod, "fetch_contracts", lambda t, sym, n_expiries, per_expiry_cap: cdf)
    monkeypatch.setattr(app_mod, "select_symbols", lambda *a, **k: ["X1","X2"])
    monkeypatch.setattr(app_mod, "snapshot", lambda *a, **k: {})
    # prepare_df will raise "No greeks available ..."
    def _raise(*args, **kwargs): raise RuntimeError("No greeks available (even after fallback).")
    monkeypatch.setattr(app_mod, "prepare_df", _raise)

    with pytest.raises(RuntimeError, match=r"No gamma available"):
        app_mod.main(trade=trade, opt_data=opt, stk_data=stk)


