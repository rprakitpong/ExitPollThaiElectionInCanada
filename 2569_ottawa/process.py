"""
process_votes.py
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------
# MODE SWITCH
# -----------------------
create_from_file_with_timestamp = True

# Non-deterministic shuffling
ROW_SHUFFLE_RANDOM_STATE = None

# -----------------------
# INPUT
# -----------------------
input_csv_files = [
    [
        "exitpoll_YVR_Kawin_2026-01-26.csv",
        "Vancouver",
        25,
        "https://github.com/rprakitpong/ExitPollThaiElectionInCanada/tree/main/2569_vancouver",
        20,
    ],
    [
        "exitpoll_A1_2026-01-25.csv",
        "Ottawa",
        16,
        "https://github.com/rprakitpong/ExitPollThaiElectionInCanada/tree/main/2569_ottawa",
        15,
    ],
]

# -----------------------
# ENUMS + COLORS
# -----------------------
THIRD_COL_ENUMS = [
    "PEOPLE",
    "BHUMJAITHAI",
    "DEMOCRAT",
    "PHEU_THAI",
    "THAI_SANG_THAI",
    "UTN",
    "OTHER",
    "NO_VOTE",
]

FOURTH_COL_ENUMS = ["AGREE", "DISAGREE", "NO_RIGHT", "N/A"]

REFERENDUM_STACK_ORDER_BOTTOM_TO_TOP = ["N/A", "NO_RIGHT", "DISAGREE", "AGREE"]

DEFAULT_THIRD_COL_COLORS = {
    "PEOPLE": "#FF6A00",
    "PHEU_THAI": "#D62728",
    "BHUMJAITHAI": "#000080",
    "DEMOCRAT": "#87CEEB",
    "UTN": "#1A2A6C",
    "THAI_SANG_THAI": "#800080",
    "OTHER": "#808080",
    "NO_VOTE": "#D3D3D3",
}

DEFAULT_FOURTH_COL_COLORS = {
    "AGREE": "#2CA02C",
    "DISAGREE": "#D62728",
    "NO_RIGHT": "#808080",
    "N/A": "#D3D3D3",
}

# -----------------------
# Helpers
# -----------------------
def make_no_timestamp_csv(df: pd.DataFrame, in_path: Path) -> Path:
    df_no_ts = df.drop(df.columns[0], axis=1)
    df_no_ts = df_no_ts.sample(frac=1, random_state=ROW_SHUFFLE_RANDOM_STATE).reset_index(drop=True)
    out = in_path.with_name(in_path.stem + "_noTimestamp" + in_path.suffix)
    df_no_ts.to_csv(out, header=False, index=False)
    return out


def derive_no_timestamp_path(in_path: Path) -> Path:
    return in_path.with_name(in_path.stem + "_noTimestamp" + in_path.suffix)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("").astype(str)
    for c in df.columns:
        df[c] = df[c].str.strip().str.upper().str.lstrip("\ufeff")
    return df


def mix_color(base_hex, target_rgb, alpha):
    b = mcolors.to_rgb(base_hex)
    return mcolors.to_hex(
        (
            (1 - alpha) * b[0] + alpha * target_rgb[0],
            (1 - alpha) * b[1] + alpha * target_rgb[1],
            (1 - alpha) * b[2] + alpha * target_rgb[2],
        )
    )


def party_gradient(base_hex):
    black = (0, 0, 0)
    white = (1, 1, 1)
    return {
        "N/A": mix_color(base_hex, black, 0.55),
        "NO_RIGHT": mix_color(base_hex, black, 0.30),
        "DISAGREE": mix_color(base_hex, white, 0.15),
        "AGREE": mix_color(base_hex, white, 0.35),
    }


def annotate_bar_totals(ax, bars, values, upper_error=None):
    """
    Put the total count on top of each bar.
    If upper_error exists, put it on top of the error bar.
    """
    for bar, val in zip(bars, values):
        top = float(val)
        if upper_error and upper_error > 0:
            top += float(upper_error)

        ax.annotate(
            str(int(val)),
            xy=(bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# -----------------------
# Plot 1 & 2
# -----------------------
def plot_enum_counts(
    df,
    col_idx,
    enums,
    color_map,
    title,
    out_path,
    x_label=None,
    upper_error=None,
):
    counts = df[col_idx].value_counts().reindex(enums, fill_value=0)
    labels = list(counts.index)
    values = counts.values.astype(float)
    colors = [color_map.get(e, "#333333") for e in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    yerr = None
    capsize = 0
    if upper_error and upper_error > 0:
        yerr = np.vstack([np.zeros_like(values), np.full_like(values, upper_error)])
        capsize = 5

    bars = ax.bar(labels, values, color=colors, yerr=yerr, capsize=capsize)
    annotate_bar_totals(ax, bars, values, upper_error=upper_error)

    max_h = values.max() + (upper_error or 0)
    ax.set_ylim(0, max_h * 1.18 if max_h > 0 else 1)

    ax.set_title(title, ha="center")
    ax.set_ylabel("Count")
    if x_label:
        ax.set_xlabel(x_label)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------
# Plot 3
# -----------------------
def plot_party_stacked_linear_break(
    df,
    party_col_idx,
    ref_col_idx,
    party_order,
    party_color_map,
    title,
    out_path,
    y_break,
):
    pivot = (
        df[[party_col_idx, ref_col_idx]]
        .rename(columns={party_col_idx: "PARTY", ref_col_idx: "REF"})
        .value_counts()
        .unstack(fill_value=0)
    )

    for ref in REFERENDUM_STACK_ORDER_BOTTOM_TO_TOP:
        if ref not in pivot.columns:
            pivot[ref] = 0
    pivot = pivot[REFERENDUM_STACK_ORDER_BOTTOM_TO_TOP]

    for p in party_order:
        if p not in pivot.index:
            pivot.loc[p] = 0
    pivot = pivot.loc[party_order]

    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    max_total = float(max(totals.max(), 0.0))
    y_break = float(y_break)

    # If nothing exceeds break, render single axis (still annotate totals)
    if max_total <= y_break:
        fig, ax = plt.subplots(figsize=(13, 9))
        bottom = np.zeros(len(party_order), dtype=float)

        for ref in REFERENDUM_STACK_ORDER_BOTTOM_TO_TOP:
            vals = pivot[ref].to_numpy(dtype=float)
            colors = [party_gradient(party_color_map[p])[ref] for p in party_order]
            ax.bar(party_order, vals, bottom=bottom, color=colors)
            bottom += vals

        # totals on top
        for i in range(len(party_order)):
            ax.annotate(
                str(int(totals[i])),
                xy=(i, totals[i]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        tick_labels = []
        for p in party_order:
            r = pivot.loc[p]
            tick_labels.append(
                "\n".join(
                    [
                        p,
                        f"AGREE: {int(r['AGREE'])}",
                        f"DISAGREE: {int(r['DISAGREE'])}",
                        f"NO_RIGHT: {int(r['NO_RIGHT'])}",
                        f"N/A: {int(r['N/A'])}",
                    ]
                )
            )

        ax.set_xticks(range(len(party_order)))
        ax.set_xticklabels(tick_labels, ha="center", fontsize=9)

        ax.set_title(title, ha="center")
        ax.set_ylabel("Count")
        ax.set_xlabel("Party")
        ax.set_ylim(0, max_total * 1.18 if max_total > 0 else 1)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Broken axis:
    tip_height = 2.0
    top_padding = 1.0
    y_top_min = max_total - tip_height
    y_top_max = max_total + top_padding

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [tip_height + top_padding, y_break]},
    )

    def draw(ax):
        bottom = np.zeros(len(party_order), dtype=float)
        for ref in REFERENDUM_STACK_ORDER_BOTTOM_TO_TOP:
            vals = pivot[ref].to_numpy(dtype=float)
            colors = [party_gradient(party_color_map[p])[ref] for p in party_order]
            ax.bar(party_order, vals, bottom=bottom, color=colors)
            bottom += vals

    draw(ax_top)
    draw(ax_bot)

    ax_bot.set_ylim(0, y_break)
    ax_top.set_ylim(y_top_min, y_top_max)

    # Broken axis styling
    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.008
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, color="k", clip_on=False)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, color="k", clip_on=False)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, color="k", clip_on=False)

    # X-axis labels with vertical numbers
    tick_labels = []
    for p in party_order:
        r = pivot.loc[p]
        tick_labels.append(
            "\n".join(
                [
                    p,
                    f"AGREE: {int(r['AGREE'])}",
                    f"DISAGREE: {int(r['DISAGREE'])}",
                    f"NO_RIGHT: {int(r['NO_RIGHT'])}",
                    f"N/A: {int(r['N/A'])}",
                ]
            )
        )

    ax_bot.set_xticks(range(len(party_order)))
    ax_bot.set_xticklabels(tick_labels, ha="center", fontsize=9)

    # Totals: annotate on the axis where the bar top is visible.
    # If total <= y_break -> label on bottom axis.
    # If total > y_break  -> label on top axis.
    for i in range(len(party_order)):
        y = float(totals[i])
        if y <= y_break:
            ax_bot.annotate(
                str(int(totals[i])),
                xy=(i, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax_top.annotate(
                str(int(totals[i])),
                xy=(i, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax_top.set_title(title, ha="center")
    ax_bot.set_ylabel("Count")
    ax_bot.set_xlabel("Party")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    for csv_file, city, err, url, y_break in input_csv_files:
        in_path = Path(csv_file)

        if create_from_file_with_timestamp:
            df_raw = clean_dataframe(pd.read_csv(in_path, header=None, dtype=str))
            df = clean_dataframe(pd.read_csv(make_no_timestamp_csv(df_raw, in_path), header=None, dtype=str))
        else:
            df = clean_dataframe(pd.read_csv(derive_no_timestamp_path(in_path), header=None, dtype=str))

        total = len(df)

        plot_enum_counts(
            df,
            1,
            THIRD_COL_ENUMS,
            DEFAULT_THIRD_COL_COLORS,
            f"Exit poll of 2569/2026 Thai general election in {city}\n"
            f"Total poll participants: {total}; Max estimated misses: {err}\n"
            f"Source: {url}",
            in_path.with_name(in_path.stem + "_party.png"),
            x_label="Party",
            upper_error=err,
        )

        plot_enum_counts(
            df,
            2,
            FOURTH_COL_ENUMS,
            DEFAULT_FOURTH_COL_COLORS,
            f"Exit poll of 2569/2026 Thai referendum in {city}\n"
            f"Total poll participants: {total}; Max estimated misses: {err}\n"
            f"Source: {url}",
            in_path.with_name(in_path.stem + "_referendum.png"),
            upper_error=err,
        )

        plot_party_stacked_linear_break(
            df,
            1,
            2,
            THIRD_COL_ENUMS,
            DEFAULT_THIRD_COL_COLORS,
            f"Exit poll of 2569/2026 Thai referendum by party-list votes in {city}\n"
            f"Total poll participants: {total}; Max estimated misses: {err} (not shown)\n"
            f"Source: {url}",
            in_path.with_name(in_path.stem + "_party_stacked_break.png"),
            y_break,
        )

        print(f"Finished plots for {city}\n")


if __name__ == "__main__":
    main()
