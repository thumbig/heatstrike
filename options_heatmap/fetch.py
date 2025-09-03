from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from alpaca.trading.requests import GetOptionContractsRequest
from .contracts import normalize_contracts_response, build_contracts_df

def fetch_contracts_balanced(
    trade_client: Any,
    base_req: GetOptionContractsRequest,
    *,
    n_expiries: int,
    per_expiry_cap: int,
    max_pages: int = 50,
    page_limit: Optional[int] = 100,
) -> List[Any]:
    """
    Page through TradingClient.get_option_contracts(), collecting up to `per_expiry_cap`
    contracts per expiry across the first `n_expiries` expirations seen. Stops early when filled.

    Returns a *list of raw contract objects/dicts* (same shape as the SDK returns), suitable
    to pass into build_contracts_df().
    """
    # Clone the request so we can mutate page_token/limit without surprising caller
    req = GetOptionContractsRequest(**{k: v for k, v in base_req.__dict__.items() if not k.startswith("_")})
    # If SDK supports limit, set it (many Alpaca SDKs do)
    if page_limit is not None and hasattr(req, "limit"):
        setattr(req, "limit", page_limit)

    page_token: Optional[str] = None
    seen_by_exp: Dict[pd.Timestamp, int] = {}
    keep: List[Any] = []
    expiries_order: List[pd.Timestamp] = []

    for _ in range(max_pages):
        if page_token and hasattr(req, "page_token"):
            setattr(req, "page_token", page_token)

        resp = trade_client.get_option_contracts(req)
        batch = normalize_contracts_response(resp)
        if not batch:
            break

        # Convert this batch to a small DataFrame to know (symbol,type,strike,expiration)
        cdf = build_contracts_df(batch)
        if cdf.empty:
            # No parsable contracts in this page—skip
            pass
        else:
            # Iterate in order; take until per-expiry cap for the first n_expiries
            for _, row in cdf.sort_values(["expiration", "strike"]).iterrows():
                exp = pd.to_datetime(row["expiration"]).normalize()
                # Record expiries seen in order
                if exp not in expiries_order:
                    expiries_order.append(exp)
                # Only consider the first n_expiries we encounter
                if len(expiries_order) > n_expiries and exp not in set(expiries_order[:n_expiries]):
                    continue

                # Initialize count
                if exp not in seen_by_exp:
                    seen_by_exp[exp] = 0

                if seen_by_exp[exp] < per_expiry_cap:
                    # Find the raw contract that corresponds to this symbol in the current page,
                    # and append the raw object (not just the symbol) so caller can rebuild cdf later if needed.
                    sym = row["symbol"]
                    # In practice batch objects contain 'symbol' attribute or key
                    # Keep the first raw entry that matches this symbol
                    for raw in batch:
                        rsym = getattr(raw, "symbol", None)
                        if rsym is None and isinstance(raw, dict):
                            rsym = raw.get("symbol") or raw.get("id") or raw.get("option_symbol")
                        if rsym == sym:
                            keep.append(raw)
                            seen_by_exp[exp] += 1
                            break

                # Early exit: if first n_expiries each reached cap, stop
                if len(expiries_order) >= n_expiries:
                    first_set = expiries_order[:n_expiries]
                    if all(seen_by_exp.get(e, 0) >= per_expiry_cap for e in first_set):
                        return keep

        # Try to get next page token from resp (SDK-specific)
        page_token = getattr(resp, "next_page_token", None) or getattr(resp, "next_token", None)
        if not page_token:
            break

    return keep

