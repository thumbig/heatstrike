import types
import os
import pandas as pd
import pytest

import noweb as app_mod

class DummyTradeClient:
    def __init__(self, contracts):
        self._contracts = contracts
    def get_option_contracts(self, req):
        return self._contracts

class DummyStockHistoricalDataClient:
    def __init__(self, price=155.0):
        self._price = price
    def get_stock_latest_trade(self, req):
        symbol = getattr(req, "symbol_or_symbols", "AAPL")
        if isinstance(symbol, (list, tuple)):
            symbol = symbol[0] if symbol else "AAPL"
        return {symbol: types.SimpleNamespace(price=self._price)}

class DummyOptionHistoricalDataClient:
    def __init__(self, snaps):
        self._snaps = snaps
    def get_option_snapshot(self, req):
        syms = getattr(req, "symbol_or_symbols", []) or []
        return {s: self._snaps.get(s) for s in syms if s in self._snaps}

class DummySnapshot:
    def __init__(self, strike, exp, gamma=None):
        self.details = types.SimpleNamespace(strike_price=strike, expiration_date=pd.to_datetime(exp))
        self.greeks = None if gamma is None else types.SimpleNamespace(gamma=gamma)

def _make_occ(underlying, yymmdd, cp, strike_thou):
    return f"{underlying}{yymmdd}{cp}{strike_thou:08d}"

@pytest.fixture
def contracts_list():
    syms = [
        _make_occ("AAPL", "270617", "C", 155000),
        _make_occ("AAPL", "270617", "P", 155000),
        _make_occ("AAPL", "270624", "C", 160000),
        _make_occ("AAPL", "270624", "P", 160000),
    ]
    return [{"symbol": s} for s in syms]

@pytest.fixture
def snaps_without_gamma(contracts_list):
    mapping = {}
    for c in contracts_list:
        sym = c["symbol"]
        y = 2000 + int(sym[4:6]); m = int(sym[6:8]); d = int(sym[8:10])
        strike = int(sym[-8:]) / 1000.0
        exp = pd.Timestamp(year=y, month=m, day=d)
        mapping[sym] = DummySnapshot(strike=strike, exp=exp, gamma=None)  # no gamma
    return mapping

def test_app_raises_when_no_gamma(monkeypatch, contracts_list, snaps_without_gamma):
    # Disable default IV so fallbacks cannot rescue the run
    monkeypatch.delenv("DEFAULT_IV", raising=False)

    trade = DummyTradeClient(contracts=contracts_list)
    stk = DummyStockHistoricalDataClient(price=155.0)
    opt = DummyOptionHistoricalDataClient(snaps=snaps_without_gamma)

    with pytest.raises(RuntimeError, match=r"No gamma available"):
        app_mod.main(trade=trade, opt_data=opt, stk_data=stk)
