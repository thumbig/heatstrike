import pandas as pd
from options_heatmap.plot import plot_mispriced_heatmap

def test_plot_mispriced_heatmap_saves(tmp_path):
    df = pd.DataFrame({
        "symbol": ["A","B","C","D"],
        "type": ["call","call","put","put"],
        "strike": [100,105,100,105],
        "expiration": pd.to_datetime(["2027-06-17","2027-06-17","2027-06-24","2027-06-24"]),
        "market_price": [5.00, 4.50, 4.20, 5.10],
        "theo_price":   [4.80, 4.40, 4.50, 5.00],
        "mispricing":   [0.20, 0.10, -0.30, 0.10],
        "mispricing_pct":[0.0417, 0.0227, -0.0667, 0.02],
    })
    out = tmp_path / "mis.png"
    fig = plot_mispriced_heatmap(df, title="Mispriced", save_path=str(out), side_by_side=True, annotate="theo", annotate_fmt=".2f", use_pct=False)
    assert fig is not None and out.exists()
