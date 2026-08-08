#!/usr/bin/env python3
"""
splice_legs.py -- rebuild out/stretching_full.txt correctly across ANY number of
extension legs.

    ./splice_legs.py RUN_DIR [RUN_DIR ...]
    ./splice_legs.py --check RUN_DIR       # report only, write nothing

WHY THIS EXISTS
---------------
`extend_run.sh` spliced correctly for ONE extension and silently wrongly for two
or more. It offset only the FINAL leg by the restart step, because that was the
only leg known to be renumbered. In fact EVERY leg after the first restarts its
`stretching.txt` step column at 1 -- felbm_gpu writes `t - restart_step` there
while `series.txt` keeps absolute iterations. So with legs 1..4 the middle legs
were treated as absolute, overlapped leg 1, and had rows silently dropped by the
"keep only what is beyond the previous leg" filter.

The symptom is not a crash: lambda values per row stay correct, but the STEP
column is wrong, so t/t_a is wrong, so any window like "the last 20% of the run"
selects the wrong rows and the reported total duration is too short.

THE FIX
-------
Each leg's offset comes from its OWN series.txt, which keeps ABSOLUTE iterations
where stretching.txt renumbers from 1. Exact, and independent of how many legs
there are or whether any was killed part-way.

Deliberately NOT parsed from the log: `felbm_gpu: restarted from ... at step N`
is written to stdout, so it lands in whatever run*.log the shell redirected to
and never in out/log.txt, which is opened later in main(). Depending on that
would tie the splice to a shell invocation rather than to the data.

Runs re-done over the same span are handled: ca1e-1 has leg2 and leg3 both
covering 60000..360000 because one attempt was killed and repeated. The later
one wins.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

# A leg shorter than this is an aborted restart, not data: drop it rather than
# refuse the whole splice when its offset cannot be read.
TRIVIAL_LEG = 2000


def leg_offset(series_path):
    """Absolute step at which this leg resumed, read from its OWN series.txt.

    series.txt keeps ABSOLUTE iterations while stretching.txt renumbers from 1,
    so the pair pins the offset exactly. A resumed leg writes its first series
    row one log_skip AFTER the restart, so

        offset = first_series_iter - log_skip

    with log_skip taken from the row spacing. The original run is the exception:
    it has a row at iteration 0, and its offset is 0.

    NOT parsed from the log: `felbm_gpu: restarted from ... at step N` goes to
    stdout, which lands in run_leg<N>.log, never in out/log.txt -- that stream is
    opened later in main(). Deriving it from series.txt avoids depending on
    whichever shell redirect happened to be used.
    """
    if not os.path.exists(series_path):
        return None
    a = np.loadtxt(series_path)
    # An EMPTY file loads as shape (0,), and a[None, :] turns that into (1, 0) --
    # which has len 1, so a `len(a) == 0` guard passes it through and a[0, 0]
    # raises IndexError. Test .size BEFORE reshaping. 2D ca3e-3 has an empty
    # leg3_series.txt from a restart killed inside one log_skip, and it would
    # have crashed the splice that extend_run.sh runs when the job finishes.
    if a.size == 0:
        return None
    if a.ndim == 1:
        a = a[None, :]
    first = int(a[0, 0])
    if first == 0:
        return 0
    if len(a) < 2:
        return None
    return first - int(round(np.median(np.diff(a[:, 0]))))


def collect(run):
    """[(name, stretching_path, offset)] in chronological order."""
    out = os.path.join(run, "out")
    legs = []
    for p in sorted(glob.glob(os.path.join(out, "leg*_stretching.txt")),
                    key=lambda p: int(re.search(r"leg(\d+)_", p).group(1))):
        n = int(re.search(r"leg(\d+)_", p).group(1))
        off = leg_offset(os.path.join(out, f"leg{n}_series.txt"))
        legs.append((f"leg{n}", p, off))
    cur = os.path.join(out, "stretching.txt")
    if os.path.exists(cur):
        legs.append(("current", cur, leg_offset(os.path.join(out, "series.txt"))))
    return legs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args()

    for run in a.runs:
        legs = collect(run)
        if not legs:
            print(f"{run}: no stretching data"); continue
        print(f"\n{run}")
        rows, head = [], None
        bad = False
        for name, path, off in legs:
            nrows = 0
            if os.path.exists(path):
                try:
                    nrows = sum(1 for l in open(path) if not l.startswith("#"))
                except OSError:
                    nrows = 0
            if off is None:
                # A leg killed inside one log_skip has <2 series rows, so its
                # offset cannot be derived -- but it also holds no data worth
                # keeping. Drop those quietly; refuse only if a SUBSTANTIAL leg
                # cannot be placed, where guessing would corrupt the record.
                if nrows < TRIVIAL_LEG:
                    print(f"  {name:8s} {nrows:>9d} rows  aborted leg, no usable offset -- dropped")
                    continue
                print(f"  {name:8s} {nrows:>9d} rows  NO OFFSET and too big to drop -- REFUSING")
                bad = True
                continue
            d = np.loadtxt(path)
            if d.ndim == 1:
                d = d[None, :]
            if head is None:
                head = [l for l in open(path) if l.startswith("#")]
            d = d.copy()
            d[:, 0] += off
            print(f"  {name:8s} {len(d):>9d} rows  offset {off:>10d}  -> steps "
                  f"{int(d[0,0]):>9d}..{int(d[-1,0]):>9d}")
            rows.append(d)
        if bad:
            print("  REFUSING to write: a leg has no series.txt and its offset is unknowable")
            continue

        allrows = np.vstack(rows)
        allrows = allrows[np.argsort(allrows[:, 0], kind="stable")]
        # A killed leg can be re-run over the same span (ca1e-1 has leg2 and leg3
        # both covering 60000..360000). Reversing before np.unique keeps the LAST
        # occurrence of each step -- the surviving continuation. np.unique already
        # returns them ascending, so do NOT reverse again afterwards.
        rev = allrows[::-1]
        _, keep = np.unique(rev[:, 0], return_index=True)
        allrows = rev[keep]

        gaps = np.diff(allrows[:, 0])
        ng = int((gaps > 1).sum())
        print(f"  -> {len(allrows)} rows, steps {int(allrows[0,0])}..{int(allrows[-1,0])}"
              + (f", {ng} gap(s)" if ng else ", contiguous"))

        if a.check:
            continue
        dest = os.path.join(run, "out", "stretching_full.txt")
        with open(dest, "w") as fh:
            fh.writelines(head or [])
            fh.write("# SPLICED by splice_legs.py: absolute steps, all legs, "
                     "offsets derived from each leg's series.txt.\n")
            np.savetxt(fh, allrows, fmt="%.10g")
        print(f"  wrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
