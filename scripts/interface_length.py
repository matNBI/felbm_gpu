#!/usr/bin/env python3
"""
interface_length.py -- specific interface length s = l_int / A_f, per Linga et al.

    ./interface_length.py RUN_DIR --d 21 [--ca 0.0846]

RUN_DIR needs out/geometry.h5 and out/output_*.h5.

WHY THIS QUANTITY
-----------------
The paper defines s = l_int / A_f, the interface length per FLUID area
(A_f = phi Lx Ly, i.e. both phases together, not one of them), and predicts

    s d ~= 2 Ca^(1/4)          Eq. (2) and the text below it

This is the sharpest cheap test of whether our morphology matches theirs. Our
lambda at Ca = 0.085 tracks their Fig. S4 curve to 3 t_a and then flattens at
0.28 where theirs decays to 0.06, and the suspicion is that our phase field
stays too finely fragmented -- which shows up directly as an s that is too high.

HOW l_int IS COMPUTED
---------------------
For a phase field with c in [0,1], integrating |grad c| across an interface
gives exactly int dc = 1 regardless of the profile shape or width. So

    l_int = sum over fluid sites of |grad c|

counts each interface once, with no thresholding and no dependence on the
chosen interface_width. In 3D the identical expression gives interfacial AREA,
so sd is comparable across 2D and 3D without changing anything here.

Solid sites have no c. Gradients therefore use the centre value in place of any
solid neighbour, i.e. a zero-gradient wall. That is not the wetting boundary
condition the solver applies, but it is the right choice for COUNTING: it stops
the grain surfaces themselves from registering as interface, which would swamp
the signal at this porosity.

THE SHORT-CIRCUIT CHECK
-----------------------
The paper notes that its two largest-Ca points behave differently because
"large clusters short-circuit the flow across the periodic boundary". That is
the same co-flowing state our old slab IC produced, and it is exactly what a
checkerboard IC is designed to avoid -- so whether we have reached it is the
question, not a detail. Reported here as the largest cluster's fraction of its
phase and whether any cluster wraps the flow direction.
"""
import argparse, glob, os, re, sys
import numpy as np
import h5py
from scipy import ndimage


def load(run, path):
    g = h5py.File(os.path.join(run, "out/geometry.h5"), "r")
    coords = g["coords"][:]
    size = tuple(int(v) for v in g["size"][:])
    with h5py.File(path, "r") as f:
        c = f["concentration"][:].astype(np.float64)
    nx, ny, nz = size
    C = np.zeros((nx, ny, nz))
    M = np.zeros((nx, ny, nz), dtype=bool)
    ix, iy, iz = coords[:, 0], coords[:, 1], coords[:, 2]
    C[ix, iy, iz] = c
    M[ix, iy, iz] = True
    return C, M, size


def grad_mag(C, M, three_d):
    """|grad c| on fluid sites; solid neighbours replaced by the centre value."""
    g2 = np.zeros_like(C)
    for ax in (0, 1, 2) if three_d else (0, 1):
        if C.shape[ax] < 2:
            continue
        fwd, bwd = np.roll(C, -1, ax), np.roll(C, 1, ax)
        mf, mb = np.roll(M, -1, ax), np.roll(M, 1, ax)
        fwd = np.where(mf, fwd, C)
        bwd = np.where(mb, bwd, C)
        # central where both sides are fluid, one-sided (halved span) otherwise
        span = np.where(mf & mb, 2.0, 1.0)
        span = np.where(~mf & ~mb, np.inf, span)
        g2 += ((fwd - bwd) / span) ** 2
    return np.sqrt(g2) * M


def clusters(C, M, three_d, axis):
    """largest-cluster fraction per phase, and whether one wraps `axis`."""
    st = ndimage.generate_binary_structure(3, 1)
    out = {}
    for name, sel in (("nw", (C >= 0.5) & M), ("w", (C < 0.5) & M)):
        lab, n = ndimage.label(sel, structure=st)
        if n == 0:
            out[name] = (0, 0.0, False)
            continue
        # merge across periodic faces
        for ax in ((0, 1, 2) if three_d else (0, 1)):
            a = np.take(lab, 0, ax)
            b = np.take(lab, lab.shape[ax] - 1, ax)
            if lab.shape[ax] < 2:
                continue
            pairs = set(zip(a[(a > 0) & (b > 0)].ravel(), b[(a > 0) & (b > 0)].ravel()))
            for p, q in pairs:
                if p != q:
                    lab[lab == q] = p
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        tot = cnt.sum()
        big = ids[cnt.argmax()]
        # does the biggest cluster touch both faces along the flow axis?
        lo = np.take(lab, 0, axis) == big
        hi = np.take(lab, lab.shape[axis] - 1, axis) == big
        out[name] = (len(ids), cnt.max() / tot, bool(lo.any() and hi.any()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--d", type=float, required=True, help="grain diameter, lu")
    ap.add_argument("--ca", type=float, default=None, help="realised Ca")
    ap.add_argument("--axis", type=int, default=None, help="flow axis (default: 1 in 2D, 2 in 3D)")
    ap.add_argument("--last", action="store_true", help="last dump only")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run, "out/output_*.h5")),
                   key=lambda p: int(re.search(r"output_(\d+)", p).group(1)))
    if not files:
        sys.exit(f"no dumps in {a.run}/out/")
    if a.last:
        files = files[-1:]

    _, _, size = load(a.run, files[0])
    three_d = size[2] > 1
    axis = a.axis if a.axis is not None else (2 if three_d else 1)
    kind = "area" if three_d else "length"

    print(f"{a.run}   {size[0]}x{size[1]}x{size[2]}   {'3D' if three_d else '2D'}, "
          f"d={a.d:g} lu, flow axis {axis}")
    if a.ca is not None:
        print(f"  paper Eq. (2):  s*d = 2 Ca^(1/4) = {2*a.ca**0.25:.3f}  at Ca = {a.ca:.3e}")
    print(f"\n{'step':>10s} {'sat':>6s} {'s*d':>7s} {'l_int/'+kind:>12s} "
          f"{'n_nw':>6s} {'big_nw':>7s} {'n_w':>6s} {'big_w':>7s}  span")
    for p in files:
        C, M, _ = load(a.run, p)
        nf = int(M.sum())
        lint = float(grad_mag(C, M, three_d).sum())
        s = lint / nf
        cl = clusters(C, M, three_d, axis)
        step = int(re.search(r"output_(\d+)", p).group(1))
        sat = float(C[M].sum()) / nf
        span = ",".join(k for k in ("nw", "w") if cl[k][2]) or "-"
        print(f"{step:10d} {sat:6.4f} {s*a.d:7.3f} {lint:12.0f} "
              f"{cl['nw'][0]:6d} {cl['nw'][1]:7.3f} {cl['w'][0]:6d} {cl['w'][1]:7.3f}  {span}")


if __name__ == "__main__":
    main()
