import pandas as pd
from options_heatmap.plot import plot_gamma_heatmap

def test_plot_gamma_heatmap_returns_axes():
    df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "strike": [100, 105, 110],
        "expiration": pd.to_datetime(["2027-06-17", "2027-06-17", "2027-06-24"]),
        "gamma": [0.01, 0.02, 0.03],
    })
    ax = plot_gamma_heatmap(df, title="Test")
    assert ax is not None
