"""Figure for the XF-C7 paper (v6, IEEE T-IV), from frozen aggregate results only.

Usage (from this folder): python make_figures.py <v2_primary_json> <tier1_json> <t3_results_json> <out_dir>
Each input is checked against the SHA-256 recorded when it was produced (pilot record, tier-1 results file, T3
results file, SAMPLE_PIN_T3 addendum 15) before anything is drawn.

fig_results.pdf: per camera, the observed mean validity time after the documented time with its 95 % interval,
  the band [lo - (5 + B_c), hi + (5 + B_c)], and the predicted mean under each convention, computed as
  theta_bar - delta_H on the same observations and weights (so a convention whose predicted mean lies outside the band
  is rejected exactly for the documented time and mid-sweep; for the others Table I has the exact intervals).
  (a) WOD, human boxes (v2.0 primary) and COCO detector (tier 1, E1); (b) AV2 replication (T3 primary: CR1 intervals
  and B^det from the T3 results file; the three-method rule's exact rejections are in Table I).
  v5 (5 Nov 2026): panel (b) moves from T2's 50 logs (four cameras) to T3's 407 logs (seven cameras).
  v6 (6 Nov 2026, IEEE T-IV, printed in gray): IEEEtran journal widths; the documented time is a black line and
  mid-sweep a dashed gray line; LiDAR A and B get different shapes at equal size; panel (a) labels each row H (human
  boxes) or D (detector); a row or camera label is bold where the documented time is rejected in the primary run
  (three methods in AV2) and carries a dagger where the camera was predicted unable to decide; convention markers sit
  higher above the observed means; the figure is taller.
  TruckScenes (T2 descriptive) is in the text since the review round of 2 Nov.
The WOD example image of the held v3 letter was dropped from v4 on 2 Nov 2026 (length); this script therefore draws no
Waymo image extract. The v3 script (../paper/make_figures.py) keeps that function.
Style: as the v3 figures (Arial 5-7 pt, TrueType 42, no gridlines, vector PDF drawn at the printed width, no creation
metadata). Palette: Wong (2011) subset #0072B2, #D55E00, #009E73, #E69F00, checked with the dataviz validator on
2026-11-01 (all pairs, light mode: pass; #E69F00 below 3:1 contrast, relieved by the legend and Table I). Shapes,
fills and line styles carry identity without colour, so the figure reads in gray print.
"""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

INK, MUTED, BAND, WHITE = "#000000", "#4d4d4d", "#e3e3e3", "#ffffff"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GRAYLINE = "#808080"
META = {"CreationDate": None, "Creator": None, "Producer": None}
COLW = 252.0 / 72.27  # IEEEtran v1.8b [journal] \columnwidth, inches (measured 6 Nov 2026)
TEXTW = 516.0 / 72.27  # IEEEtran v1.8b [journal] \textwidth, inches
PINNED = {"v2": "84614460cb374e26e765ecf75a4711755cb38c6786f448ef33905834ff5f5f1e",
          "t1": "634709cb57cef942230037a24ecc1c2381b27925cb66467d6d05041c88e98f25",
          "t3": "e28f472dca12f9438bc9d7fe7bc4cd95d3cbe0ff488a1dcfd388b8ae94d6f3dc"}
WOD_CAMS = [("FRONT", "Front"), ("FRONT_LEFT", "Front left"), ("FRONT_RIGHT", "Front right"), ("SIDE_LEFT", "Side left")]
WOD_DET_DAGGER = {"FRONT_RIGHT"}  # predicted unable to decide (EXT v1.0 E0: predicted half-width 59 ms); Table I's dagger
AV2_CAMS = [("ring_front_center", "Front center"), ("ring_front_left", "Front left"), ("ring_front_right", "Front right"),
            ("ring_side_left", "Side left"), ("ring_side_right", "Side right"), ("ring_rear_left", "Rear left"),
            ("ring_rear_right", "Rear right")]
TS_CAMS = [("CAMERA_RIGHT_FRONT", "Right front"), ("CAMERA_LEFT_BACK", "Left back")]
XLIM = (-40.0, 120.0)
MS = 3.4  # marker size, pt


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(p, key):
    h = sha(p)
    if h != PINNED[key]:
        raise SystemExit(f"{p}: SHA-256 {h} is not the pinned {PINNED[key]}")
    return json.load(open(p, encoding="utf-8"))


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": False,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "ps.fonttype": 42})


def save_exact(fig, out, dpi=450):
    """Saved at the canvas size, so the figure prints at its design width and type size."""
    fig.savefig(out, metadata=META, dpi=dpi)
    fig.savefig(out.replace(".pdf", ".png"), dpi=dpi)


def _band(ax, y, d, tol, half):
    """Grey band [lo - tol, hi + tol] around an observed interval; where it is cut at the axis, an arrowhead marks it."""
    lo, hi = d["lo"] - tol, d["hi"] + tol
    a, b = max(lo, XLIM[0]), min(hi, XLIM[1])
    ax.add_patch(Rectangle((a, y - half), b - a, 2 * half, facecolor=BAND, edgecolor="none", zorder=1))
    span = XLIM[1] - XLIM[0]
    for edge, side in ((lo < XLIM[0], -1), (hi > XLIM[1], 1)):
        if edge:
            x0 = XLIM[0] if side < 0 else XLIM[1]
            ax.add_patch(plt.Polygon([(x0, y), (x0 - side * 0.022 * span, y - half * 1.35), (x0 - side * 0.022 * span, y + half * 1.35)],
                                     closed=True, facecolor=MUTED, edgecolor="none", zorder=1.5))


def _observed(ax, y, d, filled):
    ax.plot([max(d["lo"], XLIM[0]), min(d["hi"], XLIM[1])], [y, y], color=INK, lw=0.9, zorder=5,
            solid_capstyle="butt")
    ax.plot([d["est"]], [y], marker="o", ms=MS, mfc=INK if filled else WHITE, mec=INK, mew=0.7, ls="none", zorder=6)


def _pred(ax, x, y, marker, color, filled=True):
    """A convention's predicted mean. Filled markers get a thin black edge, open ones a coloured edge, so that shape
    and fill, not colour, tell them apart in gray print."""
    if XLIM[0] <= x <= XLIM[1]:
        ax.plot([x], [y], marker=marker, ms=MS, mfc=color if filled else WHITE, mec=INK if filled else color,
                mew=0.35 if filled else 0.8, ls="none", zorder=7)


def _pred_handle(marker, color, filled=True):
    return plt.Line2D([], [], marker=marker, ms=MS, mfc=color if filled else WHITE, mec=INK if filled else color,
                      mew=0.35 if filled else 0.8, ls="none")


def _axis(ax, labels, ylim, title, xlabel):
    """labels: (y, text, bold) triples for the y tick labels."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_yticks([p for p, _, _ in labels])
    ax.set_yticklabels([t for _, t, _ in labels])
    for tick, (_, _, bold) in zip(ax.get_yticklabels(), labels):
        tick.set_fontweight("bold" if bold else "normal")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([-40, -20, 0, 20, 40, 60, 80, 100, 120])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=7, loc="left", pad=3)
    if xlabel:
        ax.set_xlabel(xlabel)


def _legend(ax, handles, labels, ncol):
    ax.legend(handles, labels, loc="lower left", ncol=ncol, frameon=False, handlelength=1.6, handletextpad=0.4,
              columnspacing=0.9, labelspacing=0.3, bbox_to_anchor=(-0.02, 1.07), fontsize=6.5, borderaxespad=0.0)


def av2_rows(t3):
    """Panel (b) rows, in AV2_CAMS order: the CR1 interval of delta_start (= theta_bar), the tolerance 5 + B^det, and
    the predicted means under LiDAR A and B (theta_bar - delta_H), all from the T3 primary run."""
    p = t3["primary"]
    ints = p["three_method"]["intervals"]["primary"]
    rows = []
    for c, name in AV2_CAMS:
        d = {h: ints[c][h]["cr1"] for h in ("start", "A", "B")}
        obs = d["start"]["est"]
        rows.append({"name": name, "start": d["start"], "tol": 5.0 + p["budgets_ms"][c],
                     "pred_A": obs - d["A"]["est"], "pred_B": obs - d["B"]["est"]})
    return rows


def wod_row_flags(v2, t1):
    """(camera, 'H' or 'D') -> (bold, dagger): bold where frame start is rejected in that row's primary run."""
    hum = set(v2["verdicts_primary_run"]["frame"]["rejected_in"])
    det = set(t1["primary"]["E1"]["verdicts_primary_run"]["frame"]["rejected_in"])
    out = {}
    for c, _ in WOD_CAMS:
        out[(c, "H")] = (c in hum, False)
        out[(c, "D")] = (c in det, c in WOD_DET_DAGGER)
    return out


def av2_label_flags(t3):
    """camera -> (bold, dagger): bold where all three intervals reject the sweep start in the primary run; dagger where
    the camera clears no contrast in the predicted-capacity gate (primary run), as in av2_t3_values.table_rows."""
    p = t3["primary"]
    rej = set(p["three_method"]["three"]["runs"]["primary"]["start"]["rejected_in"])
    cap = p["gate_bm"]["primary"]["capacity"]["cameras"]
    return {c: (c in rej, not any(x["decidable"] for x in cap[c].values())) for c, _ in AV2_CAMS}


def label_text(name, dagger):
    return name + ("†" if dagger else "")


def fig_results(v2, t1, t3, out):
    """v6: IEEEtran full width, 2.3 in tall; gray-safe marks; H/D row labels in (a); bold and dagger labels."""
    W, H = TEXTW, 2.30
    fig = plt.figure(figsize=(W, H))
    lab_w, gap = 0.80, 0.30
    pw = (W - 2 * lab_w - gap - 0.08) / 2
    y0f, hf = 0.36 / H, 1.36 / H
    ax_a = fig.add_axes((lab_w / W, y0f, pw / W, hf))
    ax_b = fig.add_axes(((lab_w + pw + gap + lab_w) / W, y0f, pw / W, hf))
    for ax in (ax_a, ax_b):
        ax.axvline(0.0, color=INK, lw=1.0, zorder=2)
        ax.set_xlabel("Validity time after the documented time (ms)")

    # (a) WOD: per camera, the human row (H) and the detector row (D), each with its own predicted means just above it
    e_h, b_h = v2["estimates"], v2["budgets_ms"]
    e_d, b_d = t1["primary"]["E1"]["estimates"], t1["primary"]["E1"]["budgets_ms"]
    flags = wod_row_flags(v2, t1)
    labels = []
    trans = blended_transform_factory(ax_a.transAxes, ax_a.transData)
    for i, (c, name) in enumerate(WOD_CAMS):
        y0 = len(WOD_CAMS) - 1 - i
        for y, e, b, filled, ref in ((y0 + 0.22, e_h, b_h, True, "H"), (y0 - 0.22, e_d, b_d, False, "D")):
            d = e[c]["delta"]
            _band(ax_a, y, d["frame"], 5.0 + b[c], 0.09)
            _observed(ax_a, y, d["frame"], filled)
            obs = d["frame"]["est"]
            yp = y + 0.15
            _pred(ax_a, obs - d["lidar"]["est"], yp, "s", VERM)
            _pred(ax_a, obs - d["cam"]["est"], yp, "^", GREEN, filled=False)
            _pred(ax_a, obs - d["pose"]["est"], yp, "D", ORANGE)
            bold, dagger = flags[(c, ref)]
            labels.append((y, label_text(ref, dagger), bold))
        ax_a.text(-0.075, y0, name, transform=trans, ha="right", va="center", fontsize=6.5)
    _axis(ax_a, labels, (-0.6, len(WOD_CAMS) - 0.35), "(a) Waymo Open Dataset: after the frame start", None)
    _legend(ax_a, [plt.Line2D([], [], color=INK, lw=1.0),
                   _pred_handle("s", VERM), _pred_handle("^", GREEN, filled=False), _pred_handle("D", ORANGE),
                   plt.Line2D([], [], color=INK, marker="o", ms=MS, mfc=INK, lw=0.9),
                   plt.Line2D([], [], color=INK, marker="o", ms=MS, mfc=WHITE, lw=0.9),
                   Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none")],
            ["Frame start", "LiDAR", "Capture", "Pose", "Human (H), 95%", "Detector (D), 95%",
             "± (5 + $B_c$)"], 3)

    # (b) AV2 replication: the observed row, with the predicted means under LiDAR A and B just above it; mid-sweep at 50 ms
    labels = []
    rows = av2_rows(t3)
    aflags = av2_label_flags(t3)
    for i, (r, (c, _)) in enumerate(zip(rows, AV2_CAMS)):
        y0 = len(rows) - 1 - i
        _band(ax_b, y0, r["start"], r["tol"], 0.13)
        _observed(ax_b, y0, r["start"], False)
        _pred(ax_b, r["pred_A"], y0 + 0.27, "<", VERM)
        _pred(ax_b, r["pred_B"], y0 + 0.27, ">", BLUE, filled=False)
        bold, dagger = aflags[c]
        labels.append((y0, label_text(r["name"], dagger), bold))
    ax_b.axvline(50.0, color=GRAYLINE, lw=1.0, ls=(0, (3, 1.6)), zorder=2)
    _axis(ax_b, labels, (-0.6, len(rows) - 0.35), "(b) Argoverse 2, 407 logs (detector): after the sweep start", None)
    _legend(ax_b, [plt.Line2D([], [], color=INK, lw=1.0),
                   _pred_handle("<", VERM), _pred_handle(">", BLUE, filled=False),
                   plt.Line2D([], [], color=GRAYLINE, lw=1.0, ls=(0, (3, 1.6))),
                   plt.Line2D([], [], color=INK, marker="o", ms=MS, mfc=WHITE, lw=0.9),
                   Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none")],
            ["Sweep start", "LiDAR A", "LiDAR B", "Mid-sweep", "Detector, 95%", "± (5 + $B_c$)"], 3)
    save_exact(fig, out)


def main(argv):
    if len(argv) != 5:
        raise SystemExit(__doc__)
    style()
    out_dir = argv[4]
    os.makedirs(out_dir, exist_ok=True)
    fig_results(load(argv[1], "v2"), load(argv[2], "t1"), load(argv[3], "t3"), os.path.join(out_dir, "fig_results.pdf"))


if __name__ == "__main__":
    main(sys.argv)
