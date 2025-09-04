import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Tuple

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame()
    pv = df.pivot_table(index="strike", columns="expiration", values=value_col, aggfunc="mean")
    # DESCENDING strikes (highest at top of heatmap)
    pv = pv.sort_index(axis=0, ascending=False)
    # Keep expiries sorted left->right
    if len(pv.columns) > 0:
        pv = pv.reindex(sorted(pv.columns), axis=1)
    return pv

def _align_rows(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Align rows across calls|puts; keep DESCENDING strike order
    all_rows = sorted(set(left.index.tolist()) | set(right.index.tolist()), reverse=True)
    if not all_rows:
        return left, right
    return left.reindex(all_rows), right.reindex(all_rows)

def _format_mmdd(columns) -> list:
    try:
        return [pd.to_datetime(c).strftime("%m/%d") for c in columns]
    except Exception:
        return list(columns)

def plot_gamma_heatmap(
    df: pd.DataFrame,
    title: str = "Gamma Heatmap",
    save_path: Optional[str] = None,
    side_by_side: bool = False,
    annotate: Optional[str] = "iv",     # None | "iv" | "symbol"
    annotate_fmt: str = ".1f",
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
            ann_c = pv_iv_c * 100 if not pv_iv_c.empty else None
            ann_p = pv_iv_p * 100 if not pv_iv_p.empty else None
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            ann_c = first_symbol_pivot(calls)
            ann_p = first_symbol_pivot(puts)

        if pv_g_c.empty:
            pv_g_c = pd.DataFrame(index=pv_g_p.index)
        if pv_g_p.empty:
            pv_g_p = pd.DataFrame(index=pv_g_c.index)
        pv_g_c, pv_g_p = _align_rows(pv_g_c, pv_g_p)

        if ann_c is not None:
            ann_c = ann_c.reindex(index=pv_g_c.index, columns=pv_g_c.columns)
        if ann_p is not None:
            ann_p = ann_p.reindex(index=pv_g_p.index, columns=pv_g_p.columns)

        xlabels_c = _format_mmdd(pv_g_c.columns)
        xlabels_p = _format_mmdd(pv_g_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        sns.heatmap(pv_g_c, ax=axes[0], cmap="coolwarm", center=0,
                    annot=False if ann_c is None else ann_c.values,
                    fmt=("" if annotate == "symbol" else annotate_fmt),
                    cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)

        sns.heatmap(pv_g_p, ax=axes[1], cmap="coolwarm", center=0,
                    annot=False if ann_p is None else ann_p.values,
                    fmt=("" if annotate == "symbol" else annotate_fmt),
                    cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)

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
        ann = pv_iv * 100 if not pv_iv.empty else None
        if ann is not None:
            ann = ann.reindex(index=pv_g.index, columns=pv_g.columns)
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
        ann = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = ann.reindex(index=pv_g.index, columns=pv_g.columns)

    xlabels = _format_mmdd(pv_g.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(pv_g, ax=ax, cmap="coolwarm", center=0,
                annot=False if ann is None else ann.values,
                fmt=("" if annotate == "symbol" else annotate_fmt),
                cbar_kws={"label": "Gamma"}, annot_kws={"fontsize": 8})
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
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
):
    """
    Same API as plot_gamma_heatmap but colors by THETA and uses a different diverging colormap.
    """
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
            pv_iv_c = _pivot(calls, "iv"); ann_c = pv_iv_c * 100 if not pv_iv_c.empty else None
            pv_iv_p = _pivot(puts,  "iv"); ann_p = pv_iv_p * 100 if not pv_iv_p.empty else None
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            ann_c = first_symbol_pivot(calls); ann_p = first_symbol_pivot(puts)

        if pv_t_c.empty: pv_t_c = pd.DataFrame(index=pv_t_p.index)
        if pv_t_p.empty: pv_t_p = pd.DataFrame(index=pv_t_c.index)
        pv_t_c, pv_t_p = _align_rows(pv_t_c, pv_t_p)

        if ann_c is not None: ann_c = ann_c.reindex(index=pv_t_c.index, columns=pv_t_c.columns)
        if ann_p is not None: ann_p = ann_p.reindex(index=pv_t_p.index, columns=pv_t_p.columns)

        xlabels_c = _format_mmdd(pv_t_c.columns)
        xlabels_p = _format_mmdd(pv_t_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        sns.heatmap(pv_t_c, ax=axes[0], cmap="RdBu_r", center=0,
                    annot=False if ann_c is None else ann_c.values,
                    fmt=("" if annotate == "symbol" else annotate_fmt),
                    cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)

        sns.heatmap(pv_t_p, ax=axes[1], cmap="RdBu_r", center=0,
                    annot=False if ann_p is None else ann_p.values,
                    fmt=("" if annotate == "symbol" else annotate_fmt),
                    cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pv_t = _pivot(df, "theta")
    if pv_t.empty:
        return None

    ann = None
    if annotate == "iv" and "iv" in df.columns:
        pv_iv = _pivot(df, "iv")
        ann = pv_iv * 100 if not pv_iv.empty else None
        if ann is not None:
            ann = ann.reindex(index=pv_t.index, columns=pv_t.columns)
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
        ann = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = ann.reindex(index=pv_t.index, columns=pv_t.columns)

    xlabels = _format_mmdd(pv_t.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(pv_t, ax=ax, cmap="RdBu_r", center=0,
                annot=False if ann is None else ann.values,
                fmt=("" if annotate == "symbol" else annotate_fmt),
                cbar_kws={"label": "Theta"}, annot_kws={"fontsize": 8})
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax

def plot_mispriced_heatmap(
    df: pd.DataFrame,
    title: str = "Mispriced Options",
    save_path: Optional[str] = None,
    side_by_side: bool = True,
    annotate: str = "theo",      # "theo" or "symbol"
    annotate_fmt: str = ".2f",   # currency formatting, shown without '$'
    use_pct: bool = True,       # color by % mispricing instead of absolute
):
    """
    Colors by mispricing (market - theoretical), centered at 0.
    Annotates with theoretical price by default.
    Returns Figure (side-by-side) or Axes (single panel).
    """
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

        # annotations: theo price or symbol
        ann_c = None; ann_p = None
        if annotate == "theo" and "theo_price" in df.columns:
            ann_c = _pivot(calls, "theo_price")
            ann_p = _pivot(puts,  "theo_price")
        elif annotate == "symbol":
            def first_symbol_pivot(sub: pd.DataFrame):
                if sub.empty: return pd.DataFrame()
                tmp = sub.sort_values(["expiration", "strike"])[["strike", "expiration", "symbol"]]
                return tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
            ann_c = first_symbol_pivot(calls)
            ann_p = first_symbol_pivot(puts)

        if pv_c.empty: pv_c = pd.DataFrame(index=pv_p.index)
        if pv_p.empty: pv_p = pd.DataFrame(index=pv_c.index)
        pv_c, pv_p = _align_rows(pv_c, pv_p)

        if ann_c is not None: ann_c = ann_c.reindex(index=pv_c.index, columns=pv_c.columns)
        if ann_p is not None: ann_p = ann_p.reindex(index=pv_p.index, columns=pv_p.columns)

        xlabels_c = _format_mmdd(pv_c.columns)
        xlabels_p = _format_mmdd(pv_p.columns)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        sns.heatmap(
            pv_c, ax=axes[0], cmap="coolwarm", center=0,
            annot=False if ann_c is None else ann_c.values, fmt=("" if annotate == "symbol" else annotate_fmt),
            cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
            annot_kws={"fontsize": 8}
        )
        axes[0].set_title(f"{title} — Calls")
        axes[0].set_ylabel("Strike")
        axes[0].set_xlabel("Expiration (MM/DD)")
        axes[0].set_xticklabels(xlabels_c, rotation=0)

        sns.heatmap(
            pv_p, ax=axes[1], cmap="coolwarm", center=0,
            annot=False if ann_p is None else ann_p.values, fmt=("" if annotate == "symbol" else annotate_fmt),
            cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
            annot_kws={"fontsize": 8}
        )
        axes[1].set_title(f"{title} — Puts")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Expiration (MM/DD)")
        axes[1].set_xticklabels(xlabels_p, rotation=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # single panel
    pv = _pivot(df, color_col)
    if pv.empty:
        return None

    ann = None
    if annotate == "theo" and "theo_price" in df.columns:
        ann = _pivot(df, "theo_price")
        ann = ann.reindex(index=pv.index, columns=pv.columns)
    elif annotate == "symbol":
        tmp = df.sort_values(["expiration", "strike"])[["strike","expiration","symbol"]]
        ann = tmp.pivot_table(index="strike", columns="expiration", values="symbol", aggfunc="first")
        ann = ann.reindex(index=pv.index, columns=pv.columns)

    xlabels = _format_mmdd(pv.columns)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        pv, ax=ax, cmap="coolwarm", center=0,
        annot=False if ann is None else ann.values, fmt=("" if annotate == "symbol" else annotate_fmt),
        cbar_kws={"label": ("% Mispricing" if use_pct else "Mispricing ($)")},
        annot_kws={"fontsize": 8}
    )
    ax.set_title(title)
    ax.set_ylabel("Strike")
    ax.set_xlabel("Expiration (MM/DD)")
    ax.set_xticklabels(xlabels, rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
