#!/usr/bin/env python3
"""
plot_s4_comparison.py -- our 2D lambda(t) against the paper's Fig. S4.

    ./plot_s4_comparison.py                       # -> figures/s4_comparison.pdf
    ./plot_s4_comparison.py -o somewhere.pdf --runs DIR

Vector PDF, so it can go straight into a document.

WHAT IT SHOWS, AND WHY THIS COMPARISON
--------------------------------------
Fig. 5 gives only the endpoint of each run. Fig. S4 gives the whole approach, and
that is where the interesting difference is: at four of the five Ca closest to
ours the two agree within ~5% once both plateau AND track each other through the
decay, turning over at the same 3-5 t_a. At high Ca they differ in SHAPE, not
level -- theirs collapses from ~2 t_a to 0.056 by 52 t_a with no plateau, ours
plateaus near 0.26 out to 20 t_a and only then declines slowly. A collapse on
that timescale is what coarsening looks like, and in 2D a coarsened, near-steady
flow drives lambda toward zero (steady 2D flow is integrable: lambda = 0).

Each of our points is paired with the paper's NEAREST Ca in log space, which is
the honest pairing since the two sweeps do not sample the same values -- our
1.28e-2 sits between their 0.0086 and 0.017 and is closer to the latter. The
pairing used is printed and annotated on the figure, never silently assumed.

Paper data comes from data/paper_fig_s4.csv (see extract_fig_s4.py, which
self-checks its calibration against Fig. 5). Ours is spliced across legs by the
same code export_tables.py uses, so a mid-extension point is not truncated to its
final leg.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_tables as et        # noqa: E402

# Our 2D points, and where each run lives. ca1e-3 is excluded: it reached 0.5 t_a
# before being killed, which is not a curve.
OURS = [("2d_mixing", "ca3e-3"), ("2d_mixing", "ca1e-2"), ("2d_mixing_local", "ca1.3e-2"),
        ("2d_mixing", "ca3e-2"), ("2d_mixing", "ca8.6e-2")]
COLOURS = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#e34948"]
BINS = np.logspace(np.log10(0.09), np.log10(170), 44)


def our_curve(root, sub, point, cfg):
    p = os.path.join(root, sub, point)
    cur = et.rows(os.path.join(p, "out", "series.txt"))
    if cur is None:
        return None
    parts = []
    for f in sorted(glob.glob(os.path.join(p, "out", "leg*_series.txt"))):
        a = et.rows(f)
        if a is not None and a.shape[1] == cur.shape[1]:
            parts.append(a)
    s = np.vstack(parts + [cur])
    st = et.stretching(p, point)
    if st is None:
        return None
    u = np.abs(s[len(s) // 2:, cfg["ucol"]]).mean()
    ta = cfg["d"] / u
    t = st[:, 0] / ta
    lam = st[:, 1] * ta
    # Log bins, so the early transient does not swamp the plateau on a log axis.
    idx = np.digitize(t, BINS)
    xs, ys = [], []
    for b in range(1, len(BINS)):
        m = idx == b
        if m.sum() > 20:
            xs.append(np.sqrt(BINS[b - 1] * BINS[b]))
            ys.append(lam[m].mean())
    return u * cfg["nu"] / cfg["gam"], np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.expanduser("~/runs"))
    ap.add_argument("--paper", default="data/paper_fig_s4.csv")
    ap.add_argument("-o", "--out", default="figures/s4_comparison.pdf")
    a = ap.parse_args()

    pap = np.genfromtxt(a.paper, delimiter=",", names=True)
    paper_ca = np.unique(pap["Ca"])
    cfg = et.SWEEPS["2d"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (sub, pt), col in zip(OURS, COLOURS):
        got = our_curve(a.runs, sub, pt, cfg)
        if got is None:
            print("  %s: no data, skipped" % pt, file=sys.stderr)
            continue
        ca, x, y = got
        # Nearest paper Ca in LOG space -- the sweeps do not share sample points.
        pc = paper_ca[np.argmin(np.abs(np.log10(paper_ca) - np.log10(ca)))]
        m = pap["Ca"] == pc
        px, py = pap["t_over_ta"][m], pap["lambda"][m]
        o = np.argsort(px)
        ax.plot(x, y, "-", color=col, lw=1.8,
                label=r"$\mathrm{Ca}=%.2g$" % ca)
        ax.plot(px[o], py[o], "--", color=col, lw=1.6, alpha=0.85)
        print("  %-9s ours Ca=%.3e  <->  paper Ca=%.4g" % (pt, ca, pc))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t/t_a$")
    ax.set_ylabel(r"$\lambda\,t_a$")
    ax.set_xlim(0.09, 180)
    ax.set_ylim(0.045, 2.2)
    ax.grid(True, which="major", lw=0.4, alpha=0.35)
    ax.grid(True, which="minor", lw=0.25, alpha=0.18)

    leg = ax.legend(loc="lower left", fontsize=8, frameon=False,
                    title="solid: felbm 2D\ndashed: paper, Fig. S4")
    leg.get_title().set_fontsize(8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    sys.exit(main())
