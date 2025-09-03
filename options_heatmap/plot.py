import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    pv = df.pivot_table(index="strike", columns="expiration", values=value_col, aggfunc="mean")
    # DESC strikes (highest at top)
    pv = pv.sort_index(axis=0, ascending=False)
    # Expiries left->right
    if len(pv.columns) > 0:
        pv = pv.reindex(sorted(pv.columns), axis=1)
    return pv

def _align_rows(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Align rows across calls|puts; keep DESC order
    all_rows = sorted(set(left.index.tolist()) | set(right.index.tolist()), reverse=True)
    if not all_rows:
        return left, right
    return left.reindex(all_rows), right.reindex(all_rows)

def _format_mmdd(columns) -> list:
    try:
        return [pd.to_datetime(c).strftime("%m/%d") for c in columns]
    except Exception:
        return list(columns)

def _mask_annot_numeric_to_text(arr: Optional[pd.DataFrame], fmt: str) -> Optional[np.ndarray]:
    """
    Convert a numeric annotation DataFrame to a string array, blanking NaNs.
    fmt is used for non-blank values (e.g., '.1f' or '.2f').
    """
    if arr is None or arr.empty:
        return None
    vals = arr.values.astype(float)
    out = np.empty(vals.shape, dtype=object)
    out[:] = ""
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(vals)
    it = np.nditer(vals, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        v = it[0]
        if np.isfinite(v):
            out[idx] = f"{v:{fmt}}"
        it.iternext()
    return out

def _atm_row_index(pv: pd.DataFrame, atm_strike: Optional[float]) -> Optional[int]:
    if pv is None or pv.empty or atm_strike is None:
        return None
    # Find the row whose strike is closest to atm_strike
    diffs = np.abs(pv.index.to_numpy(dtype=float) - float(atm_strike))
    idx = int(diffs.argmin()) if len(diffs) else None
    return idx

def _draw_atm_line(ax, pv: pd.DataFrame, atm_strike: Optional[float], color="k", lw=2.0, ls="--"):
    """Draw a horizontal line across the row corresponding to the ATM strike."""
    if ax is None or pv is None or pv.empty or atm_strike is None:
        return
    idx = _atm_row_index(pv, atm_strike)
    if idx is None:
        return
    # Heatmap grid rows are centered at y = i + 0.5
    y = idx + 0.5
    ax.hlines(y, xmin=0, xmax=pv.shape[1], colors=color, linewidths=lw, linestyles=ls, zorder=10)

def plot_gamma_heatmap(
    df: pd.DataFrame,
    title: str = "Gamma Heatmap",
    save_path: Optional[str] = None,
    side_by_side: bool = False,
    annotate: Optional[str] = "iv",     # None | "iv" | "symbol"
    annotate_fmt: str = ".1f",
    atm_strike: Optional[float] = None,  # NEW
):
    if df is None or df.empty or "gamma" not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_g_c = _pivot(calls, "gamma")
        pv_g_p = _pivot(puts,  "gamma")
        if pv_g_c.empty and pv_g_p.empty:
            return None

        ann_c = None
        ann_p = None
        if annotate == "iv" and "iv" in df.columns:
            pv_iv_c = _pivot(calls, "iv")
            pv_iv_p = _pivot(puts,  "iv")
            ann_c = _mask_annot_numeric_to_text(pv_iv_c * 100.0, ".1f")
            ann_p = _mask_annot_numeric_to_text(pv_iv_p * 100.0, ".1f")
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            pv_s_c = first_symbol_pivot(calls)
            pv_s_p = first_symbol_pivot(puts)
            ann_c = pv_s_c.values if not pv_s_c.empty else None
            ann_p = pv_s_p.values if not pv_s_p.empty else None

        if pv_g_c.empty:
            pv_g_c = pd.DataFrame(index=pv_g_p.index)
        if pv_g_p.empty:
            pv_g_p = pd.DataFrame(index=pv_g_c.index)
        pv_g_c, pv_g_p = _align_rows(pv_g_c, pv_g_p)

        xlabels_c = _format_mmdd(pv_g_c.columns)
        xlabels_p = _format_mmdd(pv_g_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        sns.heatmap(pv_g_c, ax=axes[0], cmap="coolwarm", center=0,
                    annot=False if ann_c is None else ann_c,
                    fmt="",  # we preformatted text; blank for NaN
                    cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_g_c, atm_strike)

        sns.heatmap(pv_g_p, ax=axes[1], cmap="coolwarm", center=0,
                    annot=False if ann_p is None else ann_p,
                    fmt="",
                    cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_g_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv_g = _pivot(df, "gamma")
    if pv_g.empty:
        return None

    ann = None
    if annotate == "iv" and "iv" in df.columns:
        pv_iv = _pivot(df, "iv")
        ann = _mask_annot_numeric_to_text(pv_iv * 100.0, ".1f")
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
        pv_s = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = pv_s.values if not pv_s.empty else None

    xlabels = _format_mmdd(pv_g.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(pv_g, ax=ax, cmap="coolwarm", center=0,
                annot=False if ann is None else ann, fmt="",
                cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    _draw_atm_line(ax, pv_g, atm_strike)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax

def plot_theta_heatmap(
    df: pd.DataFrame,
    title: str = "Theta Heatmap",
    save_path: Optional[str] = None,
    side_by_side: bool = False,
    annotate: Optional[str] = "iv",
    annotate_fmt: str = ".1f",
    atm_strike: Optional[float] = None,  # NEW
):
    if df is None or df.empty or "theta" not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_t_c = _pivot(calls, "theta")
        pv_t_p = _pivot(puts,  "theta")
        if pv_t_c.empty and pv_t_p.empty:
            return None

        ann_c = None
        ann_p = None
        if annotate == "iv" and "iv" in df.columns:
            pv_iv_c = _pivot(calls, "iv"); ann_c = _mask_annot_numeric_to_text(pv_iv_c * 100.0, ".1f")
            pv_iv_p = _pivot(puts,  "iv"); ann_p = _mask_annot_numeric_to_text(pv_iv_p * 100.0, ".1f")
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            pv_s_c = first_symbol_pivot(calls); ann_c = pv_s_c.values if not pv_s_c.empty else None
            pv_s_p = first_symbol_pivot(puts);  ann_p = pv_s_p.values if not pv_s_p.empty else None

        if pv_t_c.empty: pv_t_c = pd.DataFrame(index=pv_t_p.index)
        if pv_t_p.empty: pv_t_p = pd.DataFrame(index=pv_t_c.index)
        pv_t_c, pv_t_p = _align_rows(pv_t_c, pv_t_p)

        xlabels_c = _format_mmdd(pv_t_c.columns)
        xlabels_p = _format_mmdd(pv_t_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        sns.heatmap(pv_t_c, ax=axes[0], cmap="RdBu_r", center=0,
                    annot=False if ann_c is None else ann_c,
                    fmt="", cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_t_c, atm_strike)

        sns.heatmap(pv_t_p, ax=axes[1], cmap="RdBu_r", center=0,
                    annot=False if ann_p is None else ann_p,
                    fmt="", cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_t_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv_t = _pivot(df, "theta")
    if pv_t.empty:
        return None

    ann = None
    if annotate == "iv" and "iv" in df.columns:
        pv_iv = _pivot(df, "iv"); ann = _mask_annot_numeric_to_text(pv_iv * 100.0, ".1f")
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
        pv_s = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = pv_s.values if not pv_s.empty else None

    xlabels = _format_mmdd(pv_t.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(pv_t, ax=ax, cmap="RdBu_r", center=0,
                annot=False if ann is None else ann, fmt="",
                cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    _draw_atm_line(ax, pv_t, atm_strike)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax

def plot_mispriced_heatmap(
    df: pd.DataFrame,
    title: str = "Mispriced Options",
    save_path: Optional[str] = None,
    side_by_side: bool = True,
    annotate: str = "theo",
    annotate_fmt: str = ".2f",
    use_pct: bool = False,
    atm_strike: Optional[float] = None,  # NEW
):
    color_col = "mispricing_pct" if use_pct else "mispricing"
    if df is None or df.empty or color_col not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_c = _pivot(calls, color_col)
        pv_p = _pivot(puts,  color_col)
        if pv_c.empty and pv_p.empty:
            return None

        # Theo price annotations, blank if NaN
        ann_c = ann_p = None
        if annotate == "theo" and "theo_price" in df.columns:
            ann_c = _mask_annot_numeric_to_text(_pivot(calls, "theo_price"), ".2f")
            ann_p = _mask_annot_numeric_to_text(_pivot(puts,  "theo_price"), ".2f")
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            pv_s_c = first_symbol_pivot(calls)
            pv_s_p = first_symbol_pivot(puts)
            ann_c = pv_s_c.values if not pv_s_c.empty else None
            ann_p = pv_s_p.values if not pv_s_p.empty else None

        if pv_c.empty: pv_c = pd.DataFrame(index=pv_p.index)
        if pv_p.empty: pv_p = pd.DataFrame(index=pv_c.index)
        pv_c, pv_p = _align_rows(pv_c, pv_p)

        xlabels_c = _format_mmdd(pv_c.columns)
        xlabels_p = _format_mmdd(pv_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        sns.heatmap(
            pv_c, ax=axes[0], cmap="coolwarm", center=0,
            annot=False if ann_c is None else ann_c, fmt="",
            cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
            annot_kws={"fontsize": 8}
        )
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_c, atm_strike)

        sns.heatmap(
            pv_p, ax=axes[1], cmap="coolwarm", center=0,
            annot=False if ann_p is None else ann_p, fmt="",
            cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
            annot_kws={"fontsize": 8}
        )
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv = _pivot(df, color_col)
    if pv.empty:
        return None

    ann = None
    if annotate == "theo" and "theo_price" in df.columns:
        ann = _mask_annot_numeric_to_text(_pivot(df, "theo_price"), ".2f")
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike","expiration","symbol"]]
        pv_s = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = pv_s.values if not pv_s.empty else None

    xlabels = _format_mmdd(pv.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        pv, ax=ax, cmap="coolwarm", center=0,
        annot=False if ann is None else ann, fmt="",
        cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
        annot_kws={"fontsize": 8}
    )
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    _draw_atm_line(ax, pv, atm_strike)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
