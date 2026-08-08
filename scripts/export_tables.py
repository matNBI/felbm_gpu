#!/usr/bin/env python3
"""
export_tables.py -- write the 2D, 3D and paper lambda(Ca) tables as CSV for R.

    ./export_tables.py                      # reads ~/runs, writes ./data/*.csv
    ./export_tables.py --runs DIR --out DIR

Three files, all plain CSV with a single header row and no comment lines, so
`read.csv` takes them without arguments:

    data/lambda_2d.csv     data/lambda_3d.csv     data/lambda_paper.csv

WHY A SCRIPT AND NOT A STATIC TABLE
-----------------------------------
Several points are still extending, and lambda at a point that has not converged
is an UPPER BOUND that falls as the run lengthens -- 2D's ca8.6e-2 went 0.274 at
15 t_a to 0.1418 at 146 t_a, a 50% drop. A table pasted into a document goes
stale silently and there is no way to tell from the numbers. Re-run this instead.

THE COLUMNS THAT MATTER FOR INTERPRETATION
------------------------------------------
`t_a_run`  how many advection times the run covers. The paper averages lambda
           over t in [30, 60] t_a (SI B.1), so anything well under 30 is not
           measuring the same quantity even if it has stopped moving.
`slope`    trend of lambda over the last 20% of the run, in % per t_a. This is
           the convergence test: 2D's ca1e-2 was at -0.01%/t_a when it settled.
`status`   converged / provisional / upper_bound, from the rule below. Read
           `upper_bound` rows as "lambda is at most this", never as a data point.

Both `lambda` and the window it is averaged over come from the last 20% of the
run, so the two are consistent by construction.

Ca IS MEASURED, NOT REQUESTED. The `point` column is the directory name, i.e.
the TARGET Ca; `Ca` is diagnosed from the mean speed actually reached. They
diverge badly at low Ca -- 2D's ca1e-3 landed at 2.55e-4, a factor of 4 low --
so always plot against `Ca`, never against the label.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

# Digitised from Fig. 5, both phases; Ca from Table S3, lambda to +-0.005.
PAPER_CA = [4.3e-4, 1.1e-3, 2.6e-3, 4.3e-3, 6.2e-3, 8.6e-3, 1.7e-2, 2.3e-2, 4.8e-2, 9.9e-2]
PAPER_LAM = [0.185, 0.233, 0.259, 0.286, 0.320, 0.329, 0.314, 0.270, 0.087, 0.058]

# sweep : (subdir, nu, sigma, grain diameter, series column holding the driven
#          velocity component -- 2D writes it in col 3, 3D in col 4)
SWEEPS = {
    "2d": dict(sub="2d_mixing", nu=0.13333, gam=0.004233, d=21.0, ucol=3,
               points=["ca8.6e-2", "ca3e-2", "ca1e-2", "ca3e-3", "ca1e-3"]),
    "3d": dict(sub="3d_mixing", nu=0.13333, gam=0.004444, d=20.0, ucol=4,
               points=["ca1e-1", "ca3e-2", "ca1e-2", "ca3e-3", "ca1e-3"]),
}

# Converged when the trend has flattened AND the run is long enough to compare
# with the paper's averaging window. A point can be flat and still wrong if it
# is flat at 8 t_a -- 3D ca3e-3 sits at +0.29%/t_a after 8.4 t_a, which says
# nothing about where it ends up.
FLAT = 0.5       # |slope| in % per t_a
LONG = 25.0      # t_a


def paper_lambda(ca):
    """Paper's lambda at an arbitrary Ca, by interpolation in log Ca."""
    return float(np.interp(np.log10(ca), np.log10(PAPER_CA), PAPER_LAM))


def analyse(root, cfg, point):
    """One sweep point -> dict of columns, or None if it has no data yet."""
    p = os.path.join(root, cfg["sub"], point)
    series = os.path.join(p, "out", "series.txt")
    if not os.path.exists(series):
        return None

    # series.txt keeps ABSOLUTE iterations across legs, so the legs just stack.
    parts = [np.loadtxt(f) for f in sorted(glob.glob(os.path.join(p, "out", "leg*_series.txt")))]
    s = np.vstack(parts + [np.loadtxt(series)])

    # stretching_full.txt is the spliced record with absolute steps. Plain
    # stretching.txt is only the LAST leg and renumbers from 1, so preferring it
    # would silently truncate every extended run to its final leg.
    full = os.path.join(p, "out", "stretching_full.txt")
    st = np.loadtxt(full if os.path.exists(full) else os.path.join(p, "out", "stretching.txt"))

    # Mean speed over the second half, so the startup transient is excluded.
    u = np.abs(s[len(s) // 2:, cfg["ucol"]]).mean()
    t_a = cfg["d"] / u
    ca = u * cfg["nu"] / cfg["gam"]

    t = st[:, 0] / t_a
    lam = st[:, 1] * t_a
    m = t >= t[-1] * 0.8
    lam_mean = float(lam[m].mean())
    slope = float(100 * np.polyfit(t[m], lam[m], 1)[0] / lam[m].mean())

    if abs(slope) < FLAT and t[-1] >= LONG:
        status = "converged"
    elif abs(slope) < FLAT:
        status = "provisional"      # flat, but too short to trust
    else:
        status = "upper_bound"      # still falling (or rising)

    pl = paper_lambda(ca)
    return dict(point=point, Ca=ca, t_a_steps=t_a, steps=int(st[-1, 0]),
                t_a_run=float(t[-1]), lam=lam_mean, slope=slope, status=status,
                paper_lambda=pl, ratio=lam_mean / pl)


def write(path, rows, fields, fmt):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt.get(k, "%s") % r[k] if k in fmt else r[k] for k in fields])
    print("wrote %s  (%d rows)" % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.expanduser("~/runs"))
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    fields = ["point", "Ca", "t_a_steps", "steps", "t_a_run", "lambda", "slope",
              "status", "paper_lambda", "ratio"]
    fmt = {"Ca": "%.6e", "t_a_steps": "%.1f", "t_a_run": "%.2f", "lambda": "%.5f",
           "slope": "%.3f", "paper_lambda": "%.4f", "ratio": "%.4f"}

    for dim, cfg in SWEEPS.items():
        rows = []
        for pt in cfg["points"]:
            r = analyse(a.runs, cfg, pt)
            if r is None:
                print("  %s/%s: no data, skipped" % (dim, pt), file=sys.stderr)
                continue
            r["lambda"] = r.pop("lam")
            rows.append(r)
        rows.sort(key=lambda r: r["Ca"])
        write(os.path.join(a.out, "lambda_%s.csv" % dim), rows, fields, fmt)

    prows = [dict(Ca=c, **{"lambda": l}) for c, l in zip(PAPER_CA, PAPER_LAM)]
    write(os.path.join(a.out, "lambda_paper.csv"), prows, ["Ca", "lambda"],
          {"Ca": "%.6e", "lambda": "%.3f"})


if __name__ == "__main__":
    sys.exit(main())
