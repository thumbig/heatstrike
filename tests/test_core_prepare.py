import pandas as pd
from options_heatmap.core import nearest_atm_strike

def test_nearest_atm_strike():
    df = pd.DataFrame({"strike":[95,100,105,110]})
    spot = 106.2
    assert nearest_atm_strike(df, spot) == 105.0
    spot2 = 99.9
    assert nearest_atm_strike(df, spot2) == 100.0

