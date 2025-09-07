import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ------------ ordering helpers ------------
def _format_mmdd(columns) -> list:
    try:
        return [pd.to_datetime(c).strftime("%m/%d") for c in columns]
    except Exception:
        return list(columns)

def _order_index(idx, *, descending: bool) -> list:
    vals = sorted(set([float(x) for x in idx]))
    return list(reversed(vals)) if descending else vals

def _pivot(df: pd.DataFrame, value_col: str, *, y_descending: bool) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    pv = df.pivot_table(index="strike", columns="expiration", values=value_col, aggfunc="mean")
    # sort expiries left->right
    if len(pv.columns) > 0:
        pv = pv.reindex(sorted(pv.columns), axis=1)
    # order strikes
    ordered = _order_index(pv.index, descending=y_descending)
    return pv.reindex(ordered)

def _align_rows(left: pd.DataFrame, right: pd.DataFrame, *, y_descending: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = sorted(set(left.index.tolist()) | set(right.index.tolist()))
    if y_descending:
        all_rows = list(reversed(all_rows))
    return left.reindex(all_rows), right.reindex(all_rows)

def _mask_annot_numeric_to_text(arr: Optional[pd.DataFrame], fmt: str) -> Optional[np.ndarray]:
    if arr is None or arr.empty:
        return None
    vals = arr.values.astype(float)
    out = np.empty(vals.shape, dtype=object)
    out[:] = ""
    it = np.nditer(vals, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        v = it[0]
        if np.isfinite(v):
            out[idx] = f"{v:{fmt}}"
        it.iternext()
    return out

def _reindex_annot_to(base: pd.DataFrame, src: Optional[pd.DataFrame], *, scale: float = 1.0, fmt: str = ".1f") -> Optional[np.ndarray]:
    if src is None or src.empty or base is None or base.empty:
        return None
    aligned = src.reindex(index=base.index, columns=base.columns)
    if scale != 1.0:
        aligned = aligned * scale
    return _mask_annot_numeric_to_text(aligned, fmt)

def _atm_row_index(pv: pd.DataFrame, atm_strike: Optional[float]) -> Optional[int]:
    if pv is None or pv.empty or atm_strike is None:
        return None
    diffs = np.abs(pv.index.to_numpy(dtype=float) - float(atm_strike))
    return int(diffs.argmin()) if len(diffs) else None

def _draw_atm_line(ax, pv: pd.DataFrame, atm_strike: Optional[float], color="k", lw=2.0, ls="--"):
    if ax is None or pv is None or pv.empty or atm_strike is None:
        return
    idx = _atm_row_index(pv, atm_strike)
    if idx is None:
        return
    y = idx + 0.5
    ax.hlines(y, xmin=0, xmax=(pv.shape[1] or 1), colors=color, linewidths=lw, linestyles=ls, zorder=10)

def _heatmap_safe(pv: pd.DataFrame, ax, *, cmap: str, center: float, annot: Optional[np.ndarray],
                  cbar_label: str, annotate_kws: dict):
    """
    Guard against zero-sized arrays (e.g., SPY path with no puts or no columns).
    If pv has no rows or no columns, skip plotting and leave the axis empty.
    """
    if pv is None or pv.empty or pv.shape[0] == 0 or pv.shape[1] == 0:
        ax.set_axis_off()
        return
    sns.heatmap(
        pv, ax=ax, cmap=cmap, center=center,
        annot=False if annot is None else annot, fmt="",
        cbar_kws={"label": cbar_label}, annot_kws=annotate_kws
    )

# ------------ seaborn heatmaps ------------
def plot_gamma_heatmap(
    df: pd.DataFrame,
    title: str = "Gamma Heatmap",
    save_path: Optional[str] = None,
    side_by_side: bool = False,
    annotate: Optional[str] = "iv",
    annotate_fmt: str = ".1f",
    atm_strike: Optional[float] = None,
    y_descending: bool = True,
):
    if df is None or df.empty or "gamma" not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_g_c = _pivot(calls, "gamma", y_descending=y_descending)
        pv_g_p = _pivot(puts,  "gamma", y_descending=y_descending)
        if pv_g_c.empty and pv_g_p.empty:
            return None

        ann_c = ann_p = None
        if annotate == "iv" and "iv" in df.columns:
            ann_c = _reindex_annot_to(pv_g_c, _pivot(calls, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_g_p, _pivot(puts,  "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame, base: pd.DataFrame):
                if sub.empty or base.empty: return None
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                pv = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
                pv = pv.reindex(index=base.index, columns=base.columns, fill_value="")
                return pv.values
            ann_c = first_symbol_pivot(calls, pv_g_c)
            ann_p = first_symbol_pivot(puts,  pv_g_p)

        pv_g_c, pv_g_p = _align_rows(pv_g_c, pv_g_p, y_descending=y_descending)
        if annotate == "iv":
            ann_c = _reindex_annot_to(pv_g_c, _pivot(calls, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_g_p, _pivot(puts,  "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)

        xlabels_c = _format_mmdd(pv_g_c.columns)
        xlabels_p = _format_mmdd(pv_g_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _heatmap_safe(pv_g_c, axes[0], cmap="coolwarm", center=0, annot=ann_c, cbar_label="Gamma", annotate_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_g_c, atm_strike)

        _heatmap_safe(pv_g_p, axes[1], cmap="coolwarm", center=0, annot=ann_p, cbar_label="Gamma", annotate_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_g_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv_g = _pivot(df, "gamma", y_descending=y_descending)
    if pv_g.empty:
        return None

    ann = None
    if annotate == "iv" and "iv" in df.columns:
        ann = _reindex_annot_to(pv_g, _pivot(df, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
        pv_s = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        pv_s = pv_s.reindex(index=pv_g.index, columns=pv_g.columns, fill_value="")
        ann = pv_s.values if not pv_s.empty else None

    xlabels = _format_mmdd(pv_g.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    _heatmap_safe(pv_g, ax, cmap="coolwarm", center=0, annot=ann, cbar_label="Gamma", annotate_kws={"fontsize": 8})
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
    atm_strike: Optional[float] = None,
    y_descending: bool = True,
):
    if df is None or df.empty or "theta" not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_t_c = _pivot(calls, "theta", y_descending=y_descending)
        pv_t_p = _pivot(puts,  "theta", y_descending=y_descending)
        if pv_t_c.empty and pv_t_p.empty:
            return None

        ann_c = ann_p = None
        if annotate == "iv" and "iv" in df.columns:
            ann_c = _reindex_annot_to(pv_t_c, _pivot(calls, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_t_p, _pivot(puts,  "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame, base: pd.DataFrame):
                if sub.empty or base.empty: return None
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                pv = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
                pv = pv.reindex(index=base.index, columns=base.columns, fill_value="")
                return pv.values
            ann_c = first_symbol_pivot(calls, pv_t_c); ann_p = first_symbol_pivot(puts, pv_t_p)

        pv_t_c, pv_t_p = _align_rows(pv_t_c, pv_t_p, y_descending=y_descending)
        if annotate == "iv":
            ann_c = _reindex_annot_to(pv_t_c, _pivot(calls, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_t_p, _pivot(puts,  "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)

        xlabels_c = _format_mmdd(pv_t_c.columns)
        xlabels_p = _format_mmdd(pv_t_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _heatmap_safe(pv_t_c, axes[0], cmap="RdBu_r", center=0, annot=ann_c, cbar_label="Theta", annotate_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_t_c, atm_strike)

        _heatmap_safe(pv_t_p, axes[1], cmap="RdBu_r", center=0, annot=ann_p, cbar_label="Theta", annotate_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_t_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv_t = _pivot(df, "theta", y_descending=y_descending)
    if pv_t.empty:
        return None

    ann = None
    if annotate == "iv" and "iv" in df.columns:
        ann = _reindex_annot_to(pv_t, _pivot(df, "iv", y_descending=y_descending), scale=100.0, fmt=annotate_fmt)

    xlabels = _format_mmdd(pv_t.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    _heatmap_safe(pv_t, ax, cmap="RdBu_r", center=0, annot=ann, cbar_label="Theta", annotate_kws={"fontsize": 8})
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
    atm_strike: Optional[float] = None,
    y_descending: bool = True,
):
    color_col = "mispricing_pct" if use_pct else "mispricing"
    if df is None or df.empty or color_col not in df.columns:
        return None

    if side_by_side and "type" in df.columns:
        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        pv_c = _pivot(calls, color_col, y_descending=y_descending)
        pv_p = _pivot(puts,  color_col, y_descending=y_descending)
        if pv_c.empty and pv_p.empty:
            return None

        ann_c = ann_p = None
        if annotate == "theo" and "theo_price" in df.columns:
            ann_c = _reindex_annot_to(pv_c, _pivot(calls, "theo_price", y_descending=y_descending), fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_p, _pivot(puts,  "theo_price", y_descending=y_descending), fmt=annotate_fmt)
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame, base: pd.DataFrame):
                if sub.empty or base.empty: return None
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                pv = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
                pv = pv.reindex(index=base.index, columns=base.columns, fill_value="")
                return pv.values
            ann_c = first_symbol_pivot(calls, pv_c); ann_p = first_symbol_pivot(puts, pv_p)

        pv_c, pv_p = _align_rows(pv_c, pv_p, y_descending=y_descending)
        if annotate == "theo":
            ann_c = _reindex_annot_to(pv_c, _pivot(calls, "theo_price", y_descending=y_descending), fmt=annotate_fmt)
            ann_p = _reindex_annot_to(pv_p, _pivot(puts,  "theo_price", y_descending=y_descending), fmt=annotate_fmt)

        xlabels_c = _format_mmdd(pv_c.columns)
        xlabels_p = _format_mmdd(pv_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        _heatmap_safe(pv_c, axes[0], cmap="coolwarm", center=0, annot=ann_c,
                      cbar_label=("% Mispricing" if use_pct else "Mispricing ($)"),
                      annotate_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)
        _draw_atm_line(axes[0], pv_c, atm_strike)

        _heatmap_safe(pv_p, axes[1], cmap="coolwarm", center=0, annot=ann_p,
                      cbar_label=("% Mispricing" if use_pct else "Mispricing ($)"),
                      annotate_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)
        _draw_atm_line(axes[1], pv_p, atm_strike)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv = _pivot(df, color_col, y_descending=y_descending)
    if pv.empty:
        return None

    ann = None
    if annotate == "theo" and "theo_price" in df.columns:
        ann = _reindex_annot_to(pv, _pivot(df, "theo_price", y_descending=y_descending), fmt=annotate_fmt)
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike","expiration","symbol"]]
        pv_s = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        pv_s = pv_s.reindex(index=pv.index, columns=pv.columns, fill_value="")
        ann = pv_s.values if not pv_s.empty else None

    xlabels = _format_mmdd(pv.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    _heatmap_safe(pv, ax, cmap="coolwarm", center=0, annot=ann,
                  cbar_label=("% Mispricing" if use_pct else "Mispricing ($)"),
                  annotate_kws={"fontsize": 8})
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    _draw_atm_line(ax, pv, atm_strike)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
