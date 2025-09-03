import re
from datetime import datetime
from typing import Tuple, Optional

_OCC_RE = re.compile(
    r"^(?P<und>[A-Z]{1,6})(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)

def parse_occ_symbol(sym: str) -> Tuple[Optional[float], Optional[datetime], Optional[str]]:
    """
    Parse OCC 21-char symbol like AAPL270617P00155000 → (155.0, 2027-06-17, "put")
    Returns (strike, expiration_datetime, type) where type ∈ {"call","put"} or Nones on failure.
    """
    if not isinstance(sym, str):
        return None, None, None
    m = _OCC_RE.match(sym)
    if not m:
        return None, None, None
    year = 2000 + int(m.group("y"))
    month = int(m.group("m"))
    day = int(m.group("d"))
    cp = m.group("cp").lower()
    strike_int = int(m.group("strike"))  # thousandths
    strike = strike_int / 1000.0
    exp = datetime(year, month, day)
    side = "call" if cp == "c" else "put"
    return strike, exp, side
