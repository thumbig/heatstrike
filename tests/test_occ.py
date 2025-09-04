from options_heatmap.occ import parse_occ_symbol
from datetime import datetime

def test_parse_occ_symbol_ok():
    s = "AAPL270617P00155000"
    strike, exp, side = parse_occ_symbol(s)
    assert strike == 155.0
    assert side == "put"
    assert isinstance(exp, datetime)
    assert exp.year == 2027 and exp.month == 6 and exp.day == 17

def test_parse_occ_symbol_bad():
    strike, exp, side = parse_occ_symbol("BAD")
    assert strike is None and exp is None and side is None
