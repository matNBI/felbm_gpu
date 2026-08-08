#!/usr/bin/env python3
"""
extract_fig_s4.py -- turn Fig. S4 of the paper into data.

    ./extract_fig_s4.py FIG.pdf -o data/paper_fig_s4.csv

Fig. S4 plots `lambda(t) t_a` against `t/t_a`, log-log, for all ten Ca of the 2D
sweep. It is the ONLY published record of how their lambda converges, which makes
it the thing to compare our runs against -- Fig. 5 gives only the endpoints.

WHY VECTOR EXTRACTION AND NOT PIXEL-READING
-------------------------------------------
The figure is a vector PDF, so the polylines are exact. Reading pixels off a
raster would add digitisation error to a comparison where the interesting
differences are tens of percent -- and it would be unverifiable. This is exact
and it self-checks (see below).

HOW IT WORKS
------------
`pdftocairo -svg` converts the page; text is emitted as outlines, which is why
no labels can be read from the file. Each of the ten Ca then appears as THREE
paths sharing one stroke colour:

    width 0.5, stroke-dasharray  -- the dotted instantaneous rate, 10^4 filaments
    width 2.0, solid, ~17 verts  -- the LOGARITHMICALLY BINNED average <- this one
    width 2.0, solid, 2 verts    -- the legend swatch

The caption names the solid line as the binned average, so that is what is taken.
The 2-vertex swatches are excluded by requiring >= 5 vertices.

CALIBRATION comes from the tick marks, not from guessing the axis limits.
Matplotlib draws majors longer than minors (3.5 vs 2.0 device units here), so the
majors give the decade spacing directly:

    x majors at 65.96, 176.70, 287.42  ->  110.73 units per decade, 1e-1 at 65.96
    y majors at 157.28, 54.19          ->  103.09 units per decade, 1e-1 at 157.28

Checked against the minors: 2e-1 predicted at 99.29 vs 99.30 measured on x, and
at 126.25 vs 126.24 on y.

SELF-CHECK, and the reason to trust the output
----------------------------------------------
Fig. S4's curves must END at the Fig. 5 values, and they do -- every one of the
ten, to 1-2%:

    Ca      4.3e-4  1.1e-3  2.6e-3  4.3e-3  6.2e-3  8.6e-3  1.7e-2  2.3e-2  4.8e-2  9.9e-2
    S4 end  0.181   0.231   0.263   0.287   0.314   0.323   0.310   0.274   0.086   0.056
    Fig. 5  0.185   0.233   0.259   0.286   0.320   0.329   0.314   0.270   0.087   0.058

`--check` prints that table and exits non-zero if any pair disagrees by more than
5%, so a future pdftocairo whose output differs will fail loudly rather than
silently produce a shifted curve.

Colour -> Ca comes from the legend order, dark to light with increasing Ca.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

# Legend order in Fig. S4: darkest is the lowest Ca.
COLOUR_CA = {
    "0%, 0%, 0%": 0.00043,
    "20.391846%, 12.940979%, 8.235168%": 0.0011,
    "40.391541%, 25.489807%, 16.078186%": 0.0026,
    "51.763916%, 32.940674%, 20.783997%": 0.0043,
    "60.391235%, 38.430786%, 24.313354%": 0.0062,
    "67.842102%, 42.744446%, 27.450562%": 0.0086,
    "83.920288%, 52.940369%, 33.724976%": 0.017,
    "90.586853%, 57.254028%, 36.470032%": 0.023,
    "100%, 67.842102%, 43.136597%": 0.048,
    "100%, 78.038025%, 49.803162%": 0.099,
}

# Fig. 5, digitised independently -- the cross-check, not an input.
FIG5 = {4.3e-4: 0.185, 1.1e-3: 0.233, 2.6e-3: 0.259, 4.3e-3: 0.286, 6.2e-3: 0.320,
        8.6e-3: 0.329, 1.7e-2: 0.314, 2.3e-2: 0.270, 4.8e-2: 0.087, 9.9e-2: 0.058}


def to_svg(pdf):
    d = tempfile.mkdtemp()
    out = os.path.join(d, "fig.svg")
    subprocess.run(["pdftocairo", "-svg", pdf, out], check=True)
    return open(out).read()


def path_points(p):
    d = re.search(r'\sd="([^"]*)"', p).group(1)
    v = re.findall(r"[ML]\s*([-\d.]+)[ ,]+([-\d.]+)", d)
    a = np.array([[float(x), float(y)] for x, y in v])
    # pdftocairo emits a y-flip on the plotted paths but not on the tick marks.
    if "matrix(1, 0, 0, -1, 0, 0)" in p:
        a[:, 1] *= -1
    return a


def calibrate(paths):
    """Decade spacing and the device position of 1e-1 on each axis, from ticks.

    Majors are longer than minors, so grouping the black 2-point segments by
    length separates them without needing to read any label.
    """
    xs, ys = [], []
    for p in paths:
        if 'stroke="rgb(0%, 0%, 0%)"' not in p:
            continue
        a = path_points(p)
        if len(a) != 2:
            continue
        dx, dy = abs(a[1, 0] - a[0, 0]), abs(a[1, 1] - a[0, 1])
        ln = float(np.hypot(dx, dy))
        if dx < 0.1 and 3.0 < ln < 4.0:
            xs.append(a[0, 0])
        if dy < 0.1 and 3.0 < ln < 4.0:
            ys.append(a[0, 1])
    xs, ys = sorted(xs), sorted(ys)
    if len(xs) < 2 or len(ys) < 2:
        sys.exit("could not find major ticks on both axes -- calibration failed")
    xd = float(np.median(np.diff(xs)))
    yd = float(np.median(np.diff(ys)))
    # x increases rightward, so the leftmost major is the smallest decade.
    # y increases DOWNWARD in SVG, so the largest y is the smallest decade.
    return xs[0], xd, ys[-1], yd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="data/paper_fig_s4.csv")
    ap.add_argument("--check", action="store_true",
                    help="print the Fig. 5 cross-check and exit non-zero on >5%% disagreement")
    a = ap.parse_args()

    svg = to_svg(a.pdf)
    paths = re.findall(r"<path[^>]*?>", svg, re.S)
    x0, xdec, y0, ydec = calibrate(paths)

    curves = {}
    for p in paths:
        c = re.search(r'stroke="rgb\(([^)]*)\)"', p)
        w = re.search(r'stroke-width="([\d.]+)"', p)
        if not c or not w or "dasharray" in p:
            continue
        if abs(float(w.group(1)) - 2.0) > 1e-6 or c.group(1) not in COLOUR_CA:
            continue
        pts = path_points(p)
        if len(pts) < 5:            # the 2-point legend swatch
            continue
        t = 0.1 * 10 ** ((pts[:, 0] - x0) / xdec)
        lam = 0.1 * 10 ** ((y0 - pts[:, 1]) / ydec)
        o = np.argsort(t)
        curves[COLOUR_CA[c.group(1)]] = (t[o], lam[o])

    if len(curves) != 10:
        print("WARNING: found %d curves, expected 10" % len(curves), file=sys.stderr)

    bad = 0
    print("%-9s %8s %8s %7s" % ("Ca", "S4 end", "Fig. 5", "ratio"))
    for ca in sorted(curves):
        end = float(curves[ca][1][-1])
        ref = FIG5.get(ca)
        r = end / ref if ref else float("nan")
        if ref and abs(r - 1) > 0.05:
            bad += 1
        print("%-9.5f %8.3f %8.3f %7.3f%s" % (ca, end, ref, r, "  <-- OFF" if ref and abs(r-1) > 0.05 else ""))
    if a.check:
        sys.exit(1 if bad else 0)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("Ca,t_over_ta,lambda\n")
        for ca in sorted(curves):
            t, lam = curves[ca]
            for ti, li in zip(t, lam):
                fh.write("%.5g,%.5f,%.5f\n" % (ca, ti, li))
    n = sum(len(v[0]) for v in curves.values())
    print("\nwrote %s  (%d curves, %d points)" % (a.out, len(curves), n))


if __name__ == "__main__":
    sys.exit(main())
