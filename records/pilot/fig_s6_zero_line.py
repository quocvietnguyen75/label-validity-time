"""S6 figure for XF-C7: the static zero line z_b (camera x viewpoint x range, collapse level 2) per decision camera.

Usage: python fig_s6_zero_line.py <zero_line_tables.parquet> <out.pdf>
Bins with fewer than 30 static observations are not drawn (they are not used by the analysis either).
Palette: ordinal blue ramp (steps 250, 400, 550, 700), validated with the dataviz validator (--ordinal, light).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CAMS = [(1, "FRONT"), (2, "FRONT_LEFT"), (3, "FRONT_RIGHT"), (4, "SIDE_LEFT")]
RANGES = ["0–15 m", "15–30 m", "30–50 m", "≥ 50 m"]
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(tables, out):
    t = pd.read_parquet(tables)
    t = t[(t["level"] == 2) & (t["n_b"] >= 30)].copy()
    t["vp_deg"] = -165.0 + 30.0 * t["viewpoint_bin"]
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.9), sharex=True, sharey=True)
    for ax, (cid, name) in zip(axes.flat, CAMS):
        g = t[t["camera"] == cid]
        ax.axhline(0.0, color=MUTED, lw=0.6, zorder=1)
        for rb, (lab, col) in enumerate(zip(RANGES, COLORS)):
            s = g[g["range_bin"] == rb].set_index("viewpoint_bin").reindex(range(12))   # gaps break the line
            s["vp_deg"] = -165.0 + 30.0 * s.index
            ax.plot(s["vp_deg"], s["z_b"], color=col, lw=1.0, marker="o", ms=2.6, mec="white", mew=0.4, label=lab, zorder=3)
        ax.set_title(name, fontsize=7, color=INK, pad=2)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-185, 185)
    fig.supxlabel("Heading relative to the line of sight (deg; 0 = seen from behind)", fontsize=7, color=INK, y=0.02)
    fig.supylabel("Static zero line $z_b$ (px)", fontsize=7, color=INK, x=0.01)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=6.5, handlelength=1.6, columnspacing=1.0,
               bbox_to_anchor=(0.53, 1.0))
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.94), h_pad=0.6, w_pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
