import pandas as pd
from options_heatmap.plot import _pivot

def test_pivot_y_descending_and_ascending():
    df = pd.DataFrame({
        "symbol": ["x"]*6,
        "strike": [100,105,110,100,105,110],
        "expiration": pd.to_datetime(["2027-06-17"]*3 + ["2027-06-24"]*3),
        "gamma": [1,2,3,4,5,6],
    })
    pv_desc = _pivot(df, "gamma", y_descending=True)
    pv_asc  = _pivot(df, "gamma", y_descending=False)
    # top index in desc should be the max strike
    assert float(pv_desc.index[0]) == 110.0
    # top index in asc should be the min strike
    assert float(pv_asc.index[0]) == 100.0

