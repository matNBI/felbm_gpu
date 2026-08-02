#!/usr/bin/env python3
"""
cluster_sizes.py -- fluid-cluster size distribution from a felbm_gpu field dump.

Answers the question that decides how low in Ca a 3D sweep can honestly go:
**is the largest fluid cluster set by the physics, or by the size of the box?**

In 2D, Linga et al. needed a 60d x 90d domain and state it was large enough to
resolve the largest clusters only for Ca >~ 1e-3; cluster area grows as
<A_c> ~ Bo^-1, so the low-Ca end is where clusters outgrow any domain. A 3D box
that fits on one GPU is ~10-13 d per side, so the same question has to be asked
directly rather than assumed.

Usage
-----
    ./cluster_sizes.py out/output_50000.h5 [--geom out/geometry.h5]
                       [--phase nw|w] [--threshold 0.5] [--d 32]
                       [--connectivity 2] [--json out.json]

Reads the compressed fluid-site ``concentration`` from the field dump and the
matching ``coords``/``size`` from geometry.h5 (written when output_xdmf = true),
rebuilds a dense grid, and labels connected components of the chosen phase with
the periodic faces merged -- the same scheme voxelize_spheres.py uses for pore
connectivity, so ``--connectivity 2`` (face+edge, 18 of 26) is what D3Q19 sees.

The verdict line is the point of the script: a cluster that touches both faces
normal to an axis is, under triple periodicity, wrapped around the domain. Its
measured size is then a lower bound imposed by the box, and any <V_c> computed
from it is meaningless. That is the signal to enlarge the domain or raise Ca.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _union_find(n):
    parent = np.arange(n + 1)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def label_periodic(mask, connectivity=2, periodic=(True, True, True)):
    """Label connected components of ``mask`` [z,y,x], merging periodic faces.

    ``periodic`` is given in (x, y, z) order to match the felbm config keys;
    the arrays are [z, y, x].  Returns (merged_labels, sizes, percolates_xyz).
    """
    from scipy import ndimage

    per = tuple(periodic)[::-1]                       # -> (z, y, x)
    st = ndimage.generate_binary_structure(3, connectivity)
    lab, nlab = ndimage.label(mask, structure=st)
    if nlab == 0:
        return lab, np.array([], dtype=np.int64), (False, False, False)

    find, union = _union_find(nlab)

    def shift(a, d, axis, is_periodic):
        if d == 0:
            return a
        if is_periodic:
            return np.roll(a, d, axis=axis)
        out = np.zeros_like(a)
        src = slice(None, -d) if d > 0 else slice(-d, None)
        dst = slice(d, None) if d > 0 else slice(None, d)
        idx_s = [slice(None)] * a.ndim; idx_s[axis] = src
        idx_d = [slice(None)] * a.ndim; idx_d[axis] = dst
        out[tuple(idx_d)] = a[tuple(idx_s)]
        return out

    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if connectivity == 2 or (dy == 0 and dx == 0)]
    for ax in range(3):
        if not per[ax]:
            continue
        plane_axes = [a for a in range(3) if a != ax]
        lo = np.take(lab, 0, axis=ax)
        hi = np.take(lab, -1, axis=ax)
        for dy, dx in offsets:
            hi_s = shift(hi, dy, 0, per[plane_axes[0]])
            hi_s = shift(hi_s, dx, 1, per[plane_axes[1]])
            m = (lo > 0) & (hi_s > 0)
            if not m.any():
                continue
            for a, b in zip(lo[m].ravel(), hi_s[m].ravel()):
                union(int(a), int(b))

    roots = np.array([find(i) for i in range(nlab + 1)])
    roots[0] = 0
    merged = roots[lab]

    ids, counts = np.unique(merged[merged > 0], return_counts=True)

    # Per-axis: does one cluster touch both bounding faces?  Under triple
    # periodicity that means it wraps the domain.
    perc = []
    for ax in range(3):
        a = set(np.unique(roots[np.take(lab, 0, axis=ax)])) - {0}
        b = set(np.unique(roots[np.take(lab, -1, axis=ax)])) - {0}
        perc.append(len(a & b) > 0)

    return merged, counts, tuple(perc[::-1])          # (x, y, z)


def spans_axes(merged, cluster_id):
    """Which axes does this specific cluster touch both faces of, in (x,y,z)."""
    out = []
    for ax in range(3):                                # ax over [z,y,x]
        lo = np.take(merged, 0, axis=ax) == cluster_id
        hi = np.take(merged, -1, axis=ax) == cluster_id
        out.append(bool(lo.any() and hi.any()))
    return tuple(out[::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("field", help="field dump, e.g. out/output_50000.h5")
    p.add_argument("--geom", default=None,
                   help="geometry.h5 (default: geometry.h5 beside the field dump)")
    p.add_argument("--phase", default="nw", choices=("nw", "w"),
                   help="which phase to cluster: nw = non-wetting (c >= threshold, "
                        "i.e. fluid 0), w = wetting (default: nw)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="concentration threshold separating the phases (default: 0.5)")
    p.add_argument("--d", type=float, default=None,
                   help="grain diameter in lattice units, to report volumes in d^3")
    p.add_argument("--connectivity", type=int, default=2, choices=(1, 2),
                   help="1 = faces only, 2 = faces+edges (D3Q19; default)")
    p.add_argument("--json", default=None, help="also write the numbers here")
    args = p.parse_args()

    import h5py

    geom = args.geom or os.path.join(os.path.dirname(args.field) or ".", "geometry.h5")
    if not os.path.isfile(geom):
        sys.exit("cluster_sizes.py: need geometry.h5 (run with output_xdmf = true); "
                 "looked for %s" % geom)

    with h5py.File(geom, "r") as h:
        coords = np.array(h["coords"])                # (N, 3) = (x, y, z)
        size = np.array(h["size"])                    # (nx, ny, nz)
    with h5py.File(args.field, "r") as h:
        if "concentration" not in h:
            sys.exit("cluster_sizes.py: no 'concentration' dataset in %s" % args.field)
        c = np.array(h["concentration"]).astype(np.float64)

    if c.size != coords.shape[0]:
        sys.exit("cluster_sizes.py: field has %d sites but geometry has %d -- "
                 "mismatched run?" % (c.size, coords.shape[0]))

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])
    sel = (c >= args.threshold) if args.phase == "nw" else (c < args.threshold)

    mask = np.zeros((nz, ny, nx), dtype=bool)
    ix, iy, iz = coords[:, 0], coords[:, 1], coords[:, 2]
    mask[iz[sel], iy[sel], ix[sel]] = True

    n_fluid = int(c.size)
    n_phase = int(sel.sum())
    merged, sizes, perc = label_periodic(mask, connectivity=args.connectivity)

    vox = 1.0 if args.d is None else (1.0 / args.d) ** 3     # voxels -> d^3
    unit = "vox" if args.d is None else "d^3"

    print("domain            : %d x %d x %d   fluid sites %d" % (nx, ny, nz, n_fluid))
    print("phase             : %s (c %s %.2f)   %d sites, saturation %.4f of pore"
          % (args.phase, ">=" if args.phase == "nw" else "<", args.threshold,
             n_phase, n_phase / float(n_fluid)))
    if sizes.size == 0:
        print("no clusters -- the phase is empty at this threshold")
        return 0

    order = np.argsort(sizes)[::-1]
    sizes_sorted = sizes[order]
    ids = np.unique(merged[merged > 0])[order]
    biggest = int(ids[0])

    print("clusters          : %d" % sizes.size)
    print("largest           : %.4g %s  (%.4f of the phase)"
          % (sizes_sorted[0] * vox, unit, sizes_sorted[0] / float(n_phase)))
    print("mean <V_c>        : %.4g %s   (volume-weighted %.4g %s)"
          % (sizes.mean() * vox, unit,
             (sizes.astype(float) ** 2).sum() / sizes.sum() * vox, unit))
    top = ", ".join("%.3g" % (s * vox) for s in sizes_sorted[:5])
    print("top 5             : %s  %s" % (top, unit))

    sp = spans_axes(merged, biggest)
    print("largest spans (x,y,z): %s" % (sp,))
    print("any cluster spans    : %s" % (perc,))

    limited = any(sp)
    print("")
    if limited:
        print("VERDICT: FINITE-SIZE LIMITED. The largest cluster wraps the periodic")
        print("         box, so its size is set by the domain, not by Ca. <V_c> from")
        print("         this run is a lower bound only -- enlarge the box or raise Ca.")
    else:
        print("VERDICT: cluster fits. The largest cluster does not touch opposite")
        print("         faces, so the domain contains it and <V_c> is meaningful.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(field=args.field, phase=args.phase,
                           nx=nx, ny=ny, nz=nz, n_fluid=n_fluid, n_phase=n_phase,
                           saturation=n_phase / float(n_fluid),
                           n_clusters=int(sizes.size),
                           largest_vox=int(sizes_sorted[0]),
                           mean_vox=float(sizes.mean()),
                           sizes_vox=[int(s) for s in sizes_sorted],
                           largest_spans_xyz=list(sp), any_spans_xyz=list(perc),
                           finite_size_limited=bool(limited)), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
