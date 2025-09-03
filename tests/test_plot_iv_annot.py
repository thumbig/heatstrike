import pandas as pd
from options_heatmap.plot import plot_gamma_heatmap

def test_plot_with_iv_annotations(tmp_path):
    df = pd.DataFrame({
        "symbol": ["A","B","C","D"],
        "strike": [100,105,100,105],
        "expiration": pd.to_datetime(["2027-06-17","2027-06-17","2027-06-24","2027-06-24"]),
        "gamma": [0.01,0.02,0.03,0.04],
        "type": ["call","call","put","put"],
        "iv": [0.20,0.22,0.25,0.27],  # 20–27%
    })
    out = tmp_path / "iv_annot.png"
    fig = plot_gamma_heatmap(df, title="Gamma", save_path=str(out), side_by_side=True, annotate="iv", annotate_fmt=".1f")
    assert fig is not None and out.exists()
