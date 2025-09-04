import pandas as pd
from types import SimpleNamespace
from options_heatmap.contracts import normalize_contracts_response, build_contracts_df

class DummyEnum:
    def __init__(self, value): self.value = value

def test_normalize_contracts_response_tuple():
    resp = ([{"symbol": "AAPL270617C00155000"}], "TOKEN")
    out = normalize_contracts_response(resp)
    assert isinstance(out, list) and out[0]["symbol"].startswith("AAPL")

def test_build_contracts_df_direct_fields():
    c = SimpleNamespace(symbol="AAPL270617C00155000",
                        type=DummyEnum("call"),
                        strike=155.0,
                        expiration_date="2027-06-17")
    df = build_contracts_df([c])
    assert list(df.columns) == ["symbol", "type", "strike", "expiration"]
    assert len(df) == 1 and df.iloc[0]["type"] == "call"

def test_build_contracts_df_occ_fallback_only():
    c = {"symbol": "AAPL270617P00155000"}
    df = build_contracts_df([c])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["type"] == "put"
    assert row["strike"] == 155.0
    assert pd.to_datetime(row["expiration"]).year == 2027
