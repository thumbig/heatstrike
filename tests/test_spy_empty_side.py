import pandas as pd
import matplotlib
matplotlib.use("Agg")

from options_heatmap.plot import plot_gamma_heatmap

def test_heatmap_handles_empty_put_side(tmp_path):
    # Simulate a SPY case where puts side ends up empty after filters
    df = pd.DataFrame({
        "symbol":["X-C-100-617","X-C-105-617"],
        "type":  ["call","call"],
        "strike":[100,105],
        "expiration": pd.to_datetime(["2027-06-17","2027-06-17"]),
        "gamma": [0.01, 0.02],
        "iv":    [0.20, 0.22],
    })
    out = tmp_path / "spy_like.png"
    fig = plot_gamma_heatmap(
        df, title="SPY-like empty puts",
        save_path=str(out), side_by_side=True,
        annotate="iv", y_descending=True
    )
    assert fig is not None
    assert out.exists()


