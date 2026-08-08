#!/usr/bin/env python3
"""
export_tables.py -- write the 2D, 3D and paper lambda(Ca) tables as CSV for R.

    ./export_tables.py                      # reads ~/runs, writes ./data/*.csv
    ./export_tables.py --runs DIR --out DIR
    ./export_tables.py --window 30 60       # the PAPER'S convention, SI B.1

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

--window LO HI: AVERAGE OVER A FIXED t/t_a WINDOW INSTEAD
---------------------------------------------------------
The default tail average reads every point at its OWN final age, which is only
sound where lambda(t) has plateaued. Where it has not, the tail average
manufactures shape: 2D ca8.6e-2 reads 0.212 over [30,60] and 0.144 by 144 t_a,
so a curve built from tail averages shows a high-Ca "collapse" that is partly
just that point having run longest.

Fig. S4 of the paper shows exactly where the distinction bites. Its curves
plateau for Ca <= 0.023 and keep decaying at 0.048 and 0.099 -- and OUR 2D
reproduces that split at the same place. Log-log slope over the last decade:

    Ca 1.06e-3  -0.105  |  6.37e-3  -0.072  |  1.28e-2  +0.023
    Ca 2.56e-2  -0.060  |  9.52e-2  -0.324   <- no plateau, as in Fig. S4

So for 2D at Ca <= 2.6e-2 the tail average is fine. For 2D at high Ca, and for
EVERY 3D point (log-log slopes -0.28 at Ca 2.9e-2 and -0.33 at 1.0e-1, i.e. a
power law with no plateau out to 150 t_a), it is not, and `--window 30 60`
reproduces the paper's own convention instead.

Note this does NOT explain the high-Ca discrepancy -- it widens it, from 2.41x
on the tail average to 3.56x on the paper's window. That was worth knowing.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splice_legs   # noqa: E402  -- offset logic lives there, not duplicated here

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


def rows(path):
    """2-D array of data rows, or None if the file is missing/empty/unreadable.

    An extension that is killed before its first log_skip leaves a leg file with
    ZERO rows -- 2D ca3e-3 has an empty leg3_series.txt from the leg2-4 restarts.
    np.loadtxt returns a zero-size array for that, which makes vstack raise
    "array at index 2 has size 0". A single-row file loads as 1-D and would break
    the same way, so both are normalised here.
    """
    if not os.path.exists(path):
        return None
    try:
        a = np.loadtxt(path)
    except (ValueError, OSError):
        return None
    if a.size == 0:
        return None
    return a[None, :] if a.ndim == 1 else a


def stretching(p, point):
    """Full stretching record with ABSOLUTE steps, across every leg.

    stretching_full.txt is written by splice_legs.py, which extend_run.sh only
    invokes AFTER the solver exits. So a point that is mid-extension has no
    stretching_full.txt at all, and plain stretching.txt is just the CURRENT leg
    renumbered from 1. Falling back to it silently reported 2D ca3e-3 -- 7.5M
    steps of history in leg1 -- as a 721k-step run at 1.13 t_a with a -36%/t_a
    slope, which is only the restart transient of the leg in flight.

    stretching_full.txt is also STALE the moment a new leg starts, so it is used
    only when it is newer than every leg file and than stretching.txt. Otherwise
    the splice is redone here, in memory, reusing splice_legs so the offset logic
    lives in exactly one place.
    """
    out = os.path.join(p, "out")
    full = os.path.join(out, "stretching_full.txt")
    sources = sorted(glob.glob(os.path.join(out, "leg*_stretching.txt")))
    cur = os.path.join(out, "stretching.txt")
    if os.path.exists(cur):
        sources.append(cur)

    if os.path.exists(full) and sources:
        if os.path.getmtime(full) >= max(os.path.getmtime(f) for f in sources):
            return rows(full)
        print("  %s: stretching_full.txt is older than a leg -- re-splicing" % point,
              file=sys.stderr)
    elif os.path.exists(full):
        return rows(full)

    legs = splice_legs.collect(p)
    parts = []
    for name, path, off in legs:
        a = rows(path)
        if a is None:
            continue
        if off is None:
            # Aborted leg: too few series rows to place it. splice_legs refuses
            # only when such a leg is SUBSTANTIAL, on the grounds that guessing
            # would corrupt the record; the same threshold applies here.
            if len(a) >= splice_legs.TRIVIAL_LEG:
                print("  %s: %s has %d rows but no derivable offset -- REFUSING"
                      % (point, name, len(a)), file=sys.stderr)
                return None
            continue
        a = a.copy()
        a[:, 0] += off
        parts.append(a)
    if not parts:
        return None
    allrows = np.vstack(parts)
    allrows = allrows[np.argsort(allrows[:, 0], kind="stable")]
    # Keep the LAST occurrence of a repeated step: a killed leg can be re-run
    # over the same span, and the survivor is the continuation.
    rev = allrows[::-1]
    _, keep = np.unique(rev[:, 0], return_index=True)
    return rev[keep]


def analyse(root, cfg, point, window=None):
    """One sweep point -> dict of columns, or None if it has no data yet."""
    p = os.path.join(root, cfg["sub"], point)
    cur = rows(os.path.join(p, "out", "series.txt"))
    if cur is None:
        return None

    # series.txt keeps ABSOLUTE iterations across legs, so the legs just stack.
    # Aborted legs are dropped rather than allowed to abort the whole point --
    # they hold no data, so nothing is lost. Legs whose column count disagrees
    # are dropped too: that means a different build wrote them, and stacking
    # them would silently misalign the velocity column.
    parts = []
    for f in sorted(glob.glob(os.path.join(p, "out", "leg*_series.txt"))):
        a = rows(f)
        if a is None:
            print("  %s: %s is empty (aborted leg), skipped" % (point, os.path.basename(f)),
                  file=sys.stderr)
            continue
        if a.shape[1] != cur.shape[1]:
            print("  %s: %s has %d columns, expected %d -- skipped"
                  % (point, os.path.basename(f), a.shape[1], cur.shape[1]), file=sys.stderr)
            continue
        parts.append(a)
    s = np.vstack(parts + [cur])

    # stretching_full.txt is the spliced record with absolute steps. Plain
    # stretching.txt is only the LAST leg and renumbers from 1, so preferring it
    # would silently truncate every extended run to its final leg.
    st = stretching(p, point)
    if st is None or len(st) < 2:
        print("  %s: no usable stretching data, skipped" % point, file=sys.stderr)
        return None

    # Mean speed over the second half, so the startup transient is excluded.
    u = np.abs(s[len(s) // 2:, cfg["ucol"]]).mean()
    t_a = cfg["d"] / u
    ca = u * cfg["nu"] / cfg["gam"]

    t = st[:, 0] / t_a
    lam = st[:, 1] * t_a

    if window is None:
        m = t >= t[-1] * 0.8
        covered = "tail20"
    else:
        lo, hi = window
        m = (t >= lo) & (t <= min(hi, t[-1]))
        # A window the run does not reach is reported as `short`, never silently
        # substituted with the tail -- that would put points averaged over
        # different ages in the same column, which is the failure this option
        # exists to prevent.
        if m.sum() < 100:
            print("  %s: reaches only %.1f t_a, window [%g,%g] unusable"
                  % (point, t[-1], lo, hi), file=sys.stderr)
            return None
        covered = "%g-%g" % (lo, min(hi, t[-1]))

    lam_mean = float(lam[m].mean())
    slope = float(100 * np.polyfit(t[m], lam[m], 1)[0] / lam[m].mean())

    # Log-log slope over the last decade: the test that distinguishes a genuine
    # plateau from a power law. d(log lam)/d(log t) -> 0 for a plateau; the
    # %/t_a slope above cannot tell them apart, because a power law t^-b has
    # relative drift -b/t and so passes ANY fixed %/t_a threshold once t is
    # large. 2D ca1e-1 measured -0.495%/t_a at 150 t_a and is a power law.
    dec = t >= max(4.0, t[-1] / 10.0)
    loglog = (float(np.polyfit(np.log(t[dec]), np.log(np.maximum(lam[dec], 1e-12)), 1)[0])
              if dec.sum() > 20 else float("nan"))

    if abs(slope) < FLAT and t[-1] >= LONG:
        status = "converged"
    elif abs(slope) < FLAT:
        status = "provisional"      # flat, but too short to trust
    else:
        status = "upper_bound"      # still falling (or rising)
    # A power law masquerades as converged on the %/t_a test. Fig. S4 of the
    # paper plateaus for Ca <= 0.023 and does not at 0.048/0.099; our 2D splits
    # in the same place, and every 3D point is on the no-plateau side.
    if loglog == loglog and loglog < -0.15:
        status = "power_law"

    pl = paper_lambda(ca)
    return dict(point=point, Ca=ca, t_a_steps=t_a, steps=int(st[-1, 0]),
                t_a_run=float(t[-1]), window=covered, lam=lam_mean, slope=slope,
                loglog=loglog, status=status, paper_lambda=pl, ratio=lam_mean / pl)


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
    ap.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                    help="average lambda over a fixed t/t_a window instead of the "
                         "last 20%%. Use 30 60 for the paper's own convention.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    fields = ["point", "Ca", "t_a_steps", "steps", "t_a_run", "window", "lambda",
              "slope", "loglog", "status", "paper_lambda", "ratio"]
    fmt = {"Ca": "%.6e", "t_a_steps": "%.1f", "t_a_run": "%.2f", "lambda": "%.5f",
           "slope": "%.3f", "loglog": "%.3f", "paper_lambda": "%.4f", "ratio": "%.4f"}
    suffix = "" if a.window is None else "_w%g-%g" % tuple(a.window)

    for dim, cfg in SWEEPS.items():
        rows = []
        for pt in cfg["points"]:
            r = analyse(a.runs, cfg, pt, a.window)
            if r is None:
                print("  %s/%s: no data, skipped" % (dim, pt), file=sys.stderr)
                continue
            r["lambda"] = r.pop("lam")
            rows.append(r)
        rows.sort(key=lambda r: r["Ca"])
        write(os.path.join(a.out, "lambda_%s%s.csv" % (dim, suffix)), rows, fields, fmt)

    prows = [dict(Ca=c, **{"lambda": l}) for c, l in zip(PAPER_CA, PAPER_LAM)]
    write(os.path.join(a.out, "lambda_paper.csv"), prows, ["Ca", "lambda"],
          {"Ca": "%.6e", "lambda": "%.3f"})


if __name__ == "__main__":
    sys.exit(main())
