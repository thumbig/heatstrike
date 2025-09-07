import pandas as pd
from options_heatmap.filters import filter_contracts_df, pick_strikes_centered_around_atm

def _mk_cdf():
    return pd.DataFrame({
        "symbol": ["S1","S2","S3","S4","S5","S6"],
        "strike": [95, 100, 105, 110, 115, 120],
        "expiration": pd.to_datetime([
            "2027-06-17","2027-06-17","2027-06-24","2027-06-24","2027-07-01","2027-07-01"
        ])
    })

def test_filter_by_strike_range():
    cdf = _mk_cdf()
    out = filter_contracts_df(cdf, strike_min=100, strike_max=110)
    assert set(out["strike"]) == {100, 105, 110}

def test_filter_by_expiry_range():
    cdf = _mk_cdf()
    out = filter_contracts_df(cdf,
                              exp_start=pd.Timestamp("2027-06-18"),
                              exp_end=pd.Timestamp("2027-06-30"))
    # Only 2027-06-24 should remain
    assert set(pd.to_datetime(out["expiration"]).dt.date) == {pd.Timestamp("2027-06-24").date()}

def test_pick_centered_strikes_odd_and_even():
    strikes = [90, 95, 100, 105, 110, 115]
    # odd count includes ATM
    out1 = pick_strikes_centered_around_atm(strikes, spot=104.5, count=3)
    assert out1 == [100.0, 105.0, 110.0]
    # even count balances around ATM (one more to the side with availability)
    out2 = pick_strikes_centered_around_atm(strikes, spot=100.1, count=4)
    assert len(out2) == 4
    assert 100.0 in out2 or 95.0 in out2

