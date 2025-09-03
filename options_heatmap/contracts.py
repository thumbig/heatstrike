from typing import Any, Iterable, List, Dict
import pandas as pd
from .occ import parse_occ_symbol

def normalize_contracts_response(contracts_resp: Any):
    """
    Accepts various SDK shapes:
      - Response object with .option_contracts
      - Tuple (list, next_page_token)
      - List-like of contracts
    Returns a concrete list.
    """
    if hasattr(contracts_resp, "option_contracts"):
        lst = contracts_resp.option_contracts
    elif isinstance(contracts_resp, tuple):
        lst, _ = contracts_resp
    else:
        lst = contracts_resp
    return list(lst or [])

def _get_attr_or_key(obj: Any, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def build_contracts_df(contracts_list: Iterable[Any]) -> pd.DataFrame:
    """
    Build a DataFrame with columns: symbol, type, strike, expiration.
    Uses direct attributes when present and falls back to parsing the OCC symbol.
    Skips entries that remain incomplete after fallback.
    """
    rows: List[Dict[str, Any]] = []
    for c in contracts_list:
        sym = (_get_attr_or_key(c, "symbol")
               or _get_attr_or_key(c, "id")
               or _get_attr_or_key(c, "option_symbol"))
        if not sym:
            # last resort: try stringifying
            s = str(c)
            for token in s.replace("/", " ").split():
                st, ex, ty = parse_occ_symbol(token)
                if st is not None:
                    sym = token
                    break
        if not sym:
            continue

        strike = (_get_attr_or_key(c, "strike")
                  or _get_attr_or_key(c, "strike_price"))
        exp = (_get_attr_or_key(c, "expiration_date")
               or _get_attr_or_key(c, "expiration"))
        c_type = (_get_attr_or_key(c, "type")
                  or _get_attr_or_key(c, "option_type"))

        if hasattr(c_type, "value"):
            c_type = c_type.value
        if isinstance(c_type, bytes):
            c_type = c_type.decode("utf-8")
        c_type = (str(c_type).lower() if c_type is not None else None)

        try:
            strike = float(strike) if strike is not None else None
        except Exception:
            strike = None

        try:
            exp = pd.to_datetime(exp) if exp is not None else None
        except Exception:
            exp = None

        if c_type not in ("call", "put"):
            c_type = None

        if strike is None or exp is None or c_type is None:
            ps, pe, pt = parse_occ_symbol(sym)
            strike = strike if strike is not None else ps
            exp = exp if exp is not None else (pd.to_datetime(pe) if pe else None)
            c_type = c_type if c_type is not None else pt

        if strike is None or exp is None or c_type not in ("call", "put"):
            continue

        rows.append(
            {"symbol": sym, "type": c_type, "strike": float(strike), "expiration": pd.to_datetime(exp)}
        )

    return pd.DataFrame(rows, columns=["symbol", "type", "strike", "expiration"])
