from typing import Optional, Literal
import math

# Normal pdf/cdf
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = ((((a5*k + a4)*k + a3)*k + a2)*k + a1) * k
    w = 1.0 - _norm_pdf(x) * poly
    return w if x >= 0 else 1.0 - w

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)

def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: Literal["call","put"]) -> float:
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    if opt_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1 = _d1(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_theta(S: float, K: float, T: float, r: float, sigma: float, opt_type: Literal["call","put"]) -> Optional[float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    term = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    carry = (-r * K * math.exp(-r * T) * _norm_cdf(d2)) if opt_type == "call" else (r * K * math.exp(-r * T) * _norm_cdf(-d2))
    return term + carry

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: Literal["call","put"]) -> Optional[float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1 = _d1(S, K, T, r, sigma)
    if opt_type == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0

def implied_vol_from_price(
    price: float, S: float, K: float, T: float, r: float, opt_type: Literal["call","put"],
    tol: float = 1e-6, max_iter: int = 100, lo: float = 1e-6, hi: float = 5.0
) -> Optional[float]:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, sig, opt_type) - price

    f_lo = f(lo); f_hi = f(hi)
    tries = 0
    while f_lo * f_hi > 0 and hi < 20.0 and tries < 20:
        hi *= 1.5
        f_hi = f(hi); tries += 1
    if f_lo * f_hi > 0:
        return None

    a, b = lo, hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or abs(b - a) < tol:
            return max(mid, lo)
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return None
