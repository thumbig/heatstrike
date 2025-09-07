import numpy as np
from options_heatmap.filters import pick_strikes_centered_around_atm

def test_numpy_array_strikes_does_not_raise():
    strikes = np.array([95, 100, 105, 110, 115], dtype=float)
    out = pick_strikes_centered_around_atm(strikes, spot=102.0, count=3)
    assert isinstance(out, list)
    assert len(out) == 3


