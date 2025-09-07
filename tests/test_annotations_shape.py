import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless

from options_heatmap.plot import plot_gamma_heatmap

def test_heatmap_iv_annotations_shape_alignment(tmp_path):
    # Uneven strikes and missing IVs simulate real-world ragged matrices.
    # Two expiries, uneven strikes; missing IVs in some cells
    df = pd.DataFrame({
        "symbol": [
            "A-C-100-617", "A-C-105-617", "A-C-110-617",
            "A-P-100-617",               "A-P-110-617",
            "A-C-100-624", "A-C-105-624",
            "A-P-100-624", "A-P-105-624", "A-P-110-624",
        ],
        "type":   ["call","call","call","put","put","call","call","put","put","put"],
        "strike": [100,   105,   110,   100,  110,  100,   105,   100,  105,  110],
        "expiration": pd.to_datetime([
            "2027-06-17","2027-06-17","2027-06-17",
            "2027-06-17","2027-06-17",
            "2027-06-24","2027-06-24",
            "2027-06-24","2027-06-24","2027-06-24",
        ]),
        "gamma":  [0.01, 0.02, 0.03, -0.01, -0.03, 0.015, 0.025, -0.012, -0.02, -0.03],
        "iv":     [0.22, 0.23, np.nan, 0.21, np.nan, np.nan, 0.24, 0.2, np.nan, 0.19],
    })

    # If annotation alignment is wrong, seaborn will raise:
    # ValueError: `data` and `annot` must have same shape.
    out_png = tmp_path / "iv_annot_shape.png"
    fig = plot_gamma_heatmap(
        df, title="Test IV Annot Shape",
        save_path=str(out_png),
        side_by_side=True,
        annotate="iv",
        annotate_fmt=".1f",
        atm_strike=105.0,
        y_descending=True,
    )
    assert fig is not None
    assert out_png.exists()
