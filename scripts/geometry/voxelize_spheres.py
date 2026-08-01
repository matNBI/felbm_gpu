#!/usr/bin/env python3
"""
voxelize_spheres.py -- periodic sphere pack -> TIFF stack for domain_geometry = 3d_image

Takes the sphere list written by ``yade_rcp.py`` (or any CSV of x,y,z,r plus a cell
size) and rasterises it onto a lattice, in exactly the layout that felbm's
``DomainInitializer_3DImage`` expects:

    * one TIFF *page* per z-slice, pages in file order;
    * within a page, row = y and column = x;
    * 8-bit, one sample per pixel, strips (not tiles) -- the reader uses
      ``TIFFReadScanline``, which cannot read tiled TIFFs;
    * grain = 255, pore = 0, to be read with ``interface_solid = true`` and
      ``invert_image = false``.

so that the C++ index ``id = col + size_x*(row + size_y*z)`` maps to voxel (x,y,z).

It also reports the geometry diagnostics that decide whether a packing is usable
for a *two-phase* run -- in particular the pore-radius distribution, since a
diffuse-interface method cannot resolve an interface in a constriction narrower
than a few interface widths.

Usage
-----
    ./voxelize_spheres.py pack/ --res 20 --out geom_d20
    ./voxelize_spheres.py pack/ --res 24 --shrink 0.03 --out geom_d24_s3
    ./voxelize_spheres.py --self-test

``--res`` is lattice units per grain *diameter*, which is the number you actually
reason about (d in lu).  ``--shrink`` reduces every radius by that fraction; in a
close packing this is usually necessary, and the paper does the analogous thing in
2D ("The initial diameter is set to d_init = 1.2 d ... we then shrink the cylinders
to a diameter d, which produces ... good connectivity of the pore space").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
#  Input
# --------------------------------------------------------------------------- #


def load_pack(path):
    """Accept either a directory written by yade_rcp.py, or a bare CSV."""
    if os.path.isdir(path):
        csv = os.path.join(path, "spheres.csv")
        meta_path = os.path.join(path, "pack.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
    else:
        csv = path
        meta = {}

    data = np.loadtxt(csv, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] != 4:
        raise ValueError("expected 4 columns (x,y,z,r) in %s, got %d"
                         % (csv, data.shape[1]))
    centres = np.ascontiguousarray(data[:, :3], dtype=np.float64)
    radii = np.ascontiguousarray(data[:, 3], dtype=np.float64)
    return centres, radii, meta


# --------------------------------------------------------------------------- #
#  Rasterisation
# --------------------------------------------------------------------------- #


def voxelize(centres, radii, cell, n):
    """Stamp spheres onto an n = (nx,ny,nz) lattice with periodic wrap.

    A voxel is solid if its *centre* lies inside a sphere.  Returns a
    ``uint8`` array indexed ``[z, y, x]`` with 255 = grain.
    """
    nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
    h = np.array([cell[0] / nx, cell[1] / ny, cell[2] / nz])

    solid = np.zeros((nz, ny, nx), dtype=bool)

    # Voxel-centre coordinates along each axis.
    for c, r in zip(centres, radii):
        if r <= 0:
            continue
        # Index range of the sphere's bounding box (unwrapped; wrapped on write).
        lo = np.floor((c - r) / h - 0.5).astype(np.int64)
        hi = np.ceil((c + r) / h - 0.5).astype(np.int64)

        ix = np.arange(lo[0], hi[0] + 1)
        iy = np.arange(lo[1], hi[1] + 1)
        iz = np.arange(lo[2], hi[2] + 1)

        dx = (ix + 0.5) * h[0] - c[0]
        dy = (iy + 0.5) * h[1] - c[1]
        dz = (iz + 0.5) * h[2] - c[2]

        m = ((dz[:, None, None] ** 2)
             + (dy[None, :, None] ** 2)
             + (dx[None, None, :] ** 2)) <= r * r
        if not m.any():
            continue

        # np.ix_ with wrapped indices does the periodic scatter in one shot.
        zz = np.mod(iz, nz)
        yy = np.mod(iy, ny)
        xx = np.mod(ix, nx)
        sub = solid[np.ix_(zz, yy, xx)]
        solid[np.ix_(zz, yy, xx)] = sub | m

    return solid


def pad_axis(solid, n_pad, axis_zyx=0):
    """Insert n_pad empty (fluid) slices at both ends of an axis of the [z,y,x] array.

    Used to reproduce the paper's 0.5 d inlet/outlet padding.  This destroys
    periodicity along that axis, so the run must use open boundaries.
    """
    if n_pad <= 0:
        return solid
    pad = [(0, 0)] * 3
    pad[axis_zyx] = (n_pad, n_pad)
    return np.pad(solid, pad, mode="constant", constant_values=False)


# --------------------------------------------------------------------------- #
#  Diagnostics
# --------------------------------------------------------------------------- #


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


def pore_connectivity(solid, periodic=(True, True, True), connectivity=2):
    """Label the pore space, merging labels across periodic faces.

    ``connectivity=2`` (face + edge neighbours, 18 of 26) matches the D3Q19
    velocity set, so it is the connectivity the solver actually sees.
    ``connectivity=1`` (faces only) is the stricter, more conservative test.

    ``periodic`` is given in (x, y, z) order; the arrays are [z, y, x].
    """
    from scipy import ndimage

    per = tuple(periodic)[::-1]                       # -> (z, y, x)

    pore = ~solid
    st = ndimage.generate_binary_structure(3, connectivity)
    lab, nlab = ndimage.label(pore, structure=st)
    if nlab == 0:
        return dict(n_clusters=0, largest_frac=0.0, isolated_frac=1.0,
                    percolates=(False, False, False))

    find, union = _union_find(nlab)

    def shift(a, d, axis, is_periodic):
        """Shift by d along `axis`: wrap if that axis is periodic, otherwise
        shift in zeros (label 0 = not pore, so no spurious union)."""
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

    # Merge across each periodic face.  Two voxels on opposite faces are
    # neighbours if their in-plane offset is within the structuring element --
    # but an in-plane offset may only wrap if that in-plane axis is itself
    # periodic, or we would join regions the solver cannot connect.
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if connectivity == 2 or (dy == 0 and dx == 0)]
    for ax in range(3):                       # ax over [z, y, x]
        if not per[ax]:
            continue
        plane_axes = [a for a in range(3) if a != ax]     # in [z,y,x] order
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
    n_pore = int(pore.sum())
    largest = int(counts.max())

    # Directional percolation, judged on the *unmerged* labels: does one cluster
    # touch both faces normal to an axis?  (For a periodic axis this is implied
    # by the merge above, but it is still the number people quote.)
    perc = []
    for ax in range(3):
        a = set(np.unique(roots[np.take(lab, 0, axis=ax)])) - {0}
        b = set(np.unique(roots[np.take(lab, -1, axis=ax)])) - {0}
        perc.append(len(a & b) > 0)

    return dict(n_clusters=int(ids.size),
                largest_frac=largest / float(n_pore),
                isolated_frac=1.0 - largest / float(n_pore),
                percolates=tuple(perc[::-1]))     # report as (x, y, z)


def _periodic_edt(mask, h, periodic, margin):
    """EDT of ``mask`` (True = the set whose distance-to-complement we want),
    honouring periodicity by working on a wrapped halo and cropping back.

    ``periodic`` is given in (x, y, z) order; the arrays are [z, y, x].
    """
    from scipy import ndimage

    per = tuple(periodic)[::-1]                       # -> (z, y, x)
    pad = [(margin, margin) if per[a] else (0, 0) for a in range(3)]
    m = np.pad(mask, pad, mode="wrap")
    # sampling is in [z, y, x] order, and h is [hx, hy, hz]
    d = ndimage.distance_transform_edt(m, sampling=(h[2], h[1], h[0]))
    sl = tuple(slice(p[0], d.shape[a] - p[1] if p[1] else None)
               for a, p in enumerate(pad))
    return d[sl]


def wall_distance(solid, h, periodic=(True, True, True), margin=None):
    """Euclidean distance from each pore voxel to the nearest grain voxel.

    Note what this is *not*: it is not the pore or throat size.  Even in a wide
    pore most voxels sit near a wall, so the volume-weighted distribution of
    wall distance is dominated by small values.  Use ``pore_size_fractions`` for
    anything that should be read as a pore or constriction width.
    """
    if margin is None:
        margin = int(math.ceil(1.0 / min(h)))     # one grain diameter of halo
    d = _periodic_edt(~solid, h, periodic, margin)
    return d[~solid]


def pore_size_fractions(solid, h, diameters, periodic=(True, True, True)):
    """Inscribed-sphere (morphological opening) pore size.

    A voxel has local pore diameter >= D if it is covered by some ball of radius
    D/2 that fits entirely inside the pore space -- the standard maximal
    inscribed sphere definition.  Returns, for each D, the fraction of pore
    volume whose local pore diameter is *below* D.

    Two distance transforms per diameter: one to find where a ball of radius R
    fits (wall distance >= R), one to find what those ball centres cover.
    """
    if len(diameters) == 0:
        return {}
    margin = int(math.ceil(max(max(diameters), 1.0 / min(h)))) + 2
    wall = _periodic_edt(~solid, h, periodic, margin)
    pore = ~solid
    n_pore = float(pore.sum())

    out = {}
    for D in diameters:
        R = 0.5 * D * h[0]                    # diameters are given in lu
        fits = wall >= R                      # ball centres that fit in the pore
        if not fits.any():
            out[D] = 1.0
            continue
        cover = _periodic_edt(~fits, h, periodic, margin) <= R
        out[D] = float((pore & ~cover).sum() / n_pore)
    return out


def report(solid, cell, h, args, meta, periodic):
    nz, ny, nx = solid.shape
    n_tot = solid.size
    n_solid = int(solid.sum())
    n_fluid = n_tot - n_solid
    poro = n_fluid / float(n_tot)

    out = {}
    print("")
    print("=" * 72)
    print("lattice        : %d x %d x %d  (%.3g voxels)" % (nx, ny, nz, n_tot))
    print("cell           : %.4f x %.4f x %.4f d" % (cell[0], cell[1], cell[2]))
    print("voxel size     : %.6f x %.6f x %.6f d   (d = %.2f/%.2f/%.2f lu)"
          % (h[0], h[1], h[2], 1 / h[0], 1 / h[1], 1 / h[2]))
    print("porosity       : %.5f      solid fraction: %.5f" % (poro, 1 - poro))
    if meta.get("porosity") is not None:
        print("               : analytic %.5f  -> discretisation error %+.2e"
              % (meta["porosity"], poro - meta["porosity"]))
    print("fluid sites    : %d" % n_fluid)
    gb = n_fluid * args.bytes_per_site / 1024.0 ** 3
    print("GPU memory est.: %.2f GB at %.0f B/site (fused path)  %s"
          % (gb, args.bytes_per_site,
             "<-- exceeds --gpu-mem" if gb > args.gpu_mem else "ok"))

    out.update(nx=nx, ny=ny, nz=nz, porosity=poro, n_fluid=n_fluid,
               mem_gb=gb)

    if not args.skip_connectivity:
        c = pore_connectivity(solid, periodic=periodic,
                              connectivity=args.connectivity)
        print("")
        print("pore connectivity (D3Q19-consistent, connectivity=%d):"
              % args.connectivity)
        print("  clusters             : %d" % c["n_clusters"])
        print("  largest cluster      : %.6f of pore volume" % c["largest_frac"])
        print("  isolated / dead pore : %.3e of pore volume  %s"
              % (c["isolated_frac"],
                 "" if c["isolated_frac"] < 1e-3 else "<-- CHECK"))
        print("  percolates (x, y, z) : %s" % (c["percolates"],))
        out["connectivity"] = c

    W = args.interface_width
    if not args.skip_edt:
        r = wall_distance(solid, h, periodic=periodic)
        pct = np.percentile(r, [1, 5, 25, 50, 75, 95]) / h[0]
        print("")
        print("wall distance (nearest grain), volume-weighted over the pore space:")
        print("  percentiles 1/5/25/50/75/95 : %s lu"
              % "  ".join("%.2f" % v for v in pct))
        print("  ... in grain diameters      : %s"
              % "  ".join("%.3f" % (v * h[0]) for v in pct))
        out["wall_distance_percentiles_lu"] = [float(v) for v in pct]

    if not args.skip_pore_size:
        ladder = [f * W for f in (1.0, 2.0, 3.0, 4.0, 6.0)]
        fr = pore_size_fractions(solid, h, ladder, periodic=periodic)
        print("")
        print("inscribed-sphere pore size vs the diffuse interface "
              "(interface_width W = %.1f lu):" % W)
        for D in ladder:
            f = fr[D]
            print("  pore volume with local diameter < %4.1f lu (=%.0f W) : %.3f"
                  % (D, D / W, f))
        crit = fr[args.interface_factor * W]
        print("  -> %.1f%% of the pore space is narrower than %.0f W, where a "
              "diffuse" % (100 * crit, args.interface_factor))
        print("     interface of this width cannot be represented. %s"
              % ("OK." if crit < 0.05 else
                 "Reduce interface_width, raise --res, or --shrink the grains."))
        out["pore_size_frac_below"] = {("%.2f" % k): v for k, v in fr.items()}
        out["frac_below_interface_scale"] = crit
    print("=" * 72)
    return out


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #


DOMAIN_CFG = """\
################################################################################
# Generated by voxelize_spheres.py -- do not hand-edit; regenerate instead.
#
#   packing        : {packing}
#   cell           : {cellstr} d   (triple-periodic: {periodic})
#   resolution     : d = {res:.2f} lattice units
#   radius shrink  : {shrink}
#   lattice        : {nx} x {ny} x {nz}
#   porosity       : {porosity:.5f}
#
# size_x/size_y/size_z in settings.cfg are OVERRIDDEN by the image dimensions.
# Use with:
#     domain_geometry = 3d_image
#     domain_cfg_file = domain.cfg
{extra_notes}################################################################################

image_dir         = {image_dir}
extension         = tif

interface_solid   = true    # voxel != 0 -> SOLID grain (we write grain = 255)
invert_image      = false   # our convention is already grain-positive

coarsening_levels = 0       # 1 halves each dimension (2x2x2 majority vote)

use_subvolume     = false
x_min             = 0
y_min             = 0
z_min             = 0
subvolume_size_x  = {nx}
subvolume_size_y  = {ny}
subvolume_size_z  = {nz}

add_inlet_pores   = false   # would destroy periodicity -- keep false
add_outlet_pores  = false
pores_pitch       = 4
pores_diameter    = 3
"""


def write_outputs(solid, outdir, cell, h, args, meta, periodic, diag):
    import tifffile

    imgdir = os.path.join(outdir, "image")
    os.makedirs(imgdir, exist_ok=True)

    vox = np.where(solid, np.uint8(255), np.uint8(0))
    tif = os.path.join(imgdir, "pack.tif")
    # Strips, not tiles: the reader uses TIFFReadScanline.
    tifffile.imwrite(tif, vox, photometric="minisblack",
                     compression=args.compression or None,
                     rowsperstrip=max(1, vox.shape[1]))

    nz, ny, nx = solid.shape
    notes = ""
    if not all(periodic):
        notes = ("#\n# NOTE: this stack is PADDED along z and is therefore NOT "
                 "periodic in z.\n#       Run it with use_open_bnd = true, "
                 "in_out_dir = 2, image_periodic = false,\n#       "
                 "transverse_periodic = true.\n")

    cfg = DOMAIN_CFG.format(
        packing=meta.get("packing", "sphere pack"),
        cellstr="%.3f x %.3f x %.3f" % tuple(cell),
        periodic=all(periodic),
        res=1.0 / h[0],
        shrink=("%.4f" % args.shrink) if args.shrink else "none",
        nx=nx, ny=ny, nz=nz,
        porosity=diag["porosity"],
        image_dir="./image/",
        extra_notes=notes,
    )
    with open(os.path.join(outdir, "domain.cfg"), "w") as f:
        f.write(cfg)

    geom = dict(source=meta, cell=list(cell), voxel=list(h),
                res_lu_per_d=1.0 / h[0], shrink=args.shrink,
                periodic=list(periodic), pad_d=args.pad,
                interface_width=args.interface_width, **diag)
    with open(os.path.join(outdir, "geometry.json"), "w") as f:
        json.dump(geom, f, indent=2, default=float)

    size_mb = os.path.getsize(tif) / 1024.0 ** 2
    print("")
    print("wrote %s  (%.1f MB, %d pages)" % (tif, size_mb, nz))
    print("      %s" % os.path.join(outdir, "domain.cfg"))
    print("      %s" % os.path.join(outdir, "geometry.json"))
    print("")
    print("In settings.cfg set:  domain_geometry = 3d_image")
    print("                      domain_cfg_file = domain.cfg")
    if all(periodic):
        print("                      image_periodic  = true")
        print("                      use_open_bnd    = false")
        print("                      empty_layers    = 0")
    return tif


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #


def lattice_pack(kind, reps):
    """Analytic packings with exactly known solid fractions, for verification.

    sc  : simple cubic,        phi = pi/6      = 0.523599
    bcc : body-centred cubic,  phi = sqrt(3)pi/8 = 0.680175
    fcc : face-centred cubic,  phi = pi/sqrt(18)  = 0.740480
    Spheres touch, so these are also the worst case for voxelised contacts.
    """
    basis = {
        "sc":  ([(0, 0, 0)], 0.5),
        "bcc": ([(0, 0, 0), (.5, .5, .5)], 0.5 * math.sqrt(3) / 2),
        "fcc": ([(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)],
                0.5 / math.sqrt(2)),
    }[kind]
    frac, r = basis
    a = 1.0                                  # lattice parameter
    cs = []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                for b in frac:
                    cs.append(((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a))
    cs = np.array(cs)
    cell = np.array([reps * a] * 3)
    phi = len(cs) * (4 / 3) * math.pi * r ** 3 / np.prod(cell)
    return cs, np.full(len(cs), r), cell, phi


def self_test():
    import tifffile
    ok = True
    print("voxelize_spheres.py self-test")
    print("-" * 72)

    # 1. Solid fraction converges to the analytic value with resolution.
    #    Centre-in-sphere sampling converges as O(h), with commensurability
    #    wobble, so we check the trend over a factor of 4 rather than each step.
    for kind, exact in (("sc", math.pi / 6),
                        ("bcc", math.sqrt(3) * math.pi / 8),
                        ("fcc", math.pi / math.sqrt(18))):
        print("%-4s exact phi = %.6f" % (kind, exact))
        errs = []
        for n in (32, 64, 128):
            cs, rs, cell, phi = lattice_pack(kind, 2)
            solid = voxelize(cs, rs, cell, (n, n, n))
            meas = solid.mean()
            err = abs(meas - exact)
            errs.append(err)
            print("     n=%3d  phi=%.6f  |err|=%.2e" % (n, meas, err))
        if errs[-1] > 1e-2:
            print("     <-- residual error too large"); ok = False
        if errs[-1] > 0.5 * errs[0]:
            print("     <-- not converging with resolution"); ok = False
        else:
            print("     error reduced %.1fx over a 4x refinement (O(h) expected: 4x)"
                  % (errs[0] / max(errs[-1], 1e-12)))

    # 2. Axis order survives a TIFF round-trip, and matches the C++ index rule.
    print("")
    print("TIFF round-trip and axis order")
    cs = np.array([[0.25, 0.25, 0.25]])
    solid = voxelize(cs, np.array([0.1]), np.array([2.0, 3.0, 4.0]), (20, 30, 40))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.tif")
        tifffile.imwrite(p, np.where(solid, np.uint8(255), np.uint8(0)),
                         photometric="minisblack", rowsperstrip=solid.shape[1])
        back = tifffile.imread(p)
    same = (back > 0).shape == solid.shape and np.array_equal(back > 0, solid)
    print("     shape (pages, rows, cols) = %s  == (nz, ny, nx) = %s : %s"
          % (back.shape, solid.shape, same))
    ok &= bool(same)
    # the sphere sits at (x,y,z) = (0.25, 0.25, 0.25) of a 2x3x4 cell
    zi, yi, xi = [int(v.mean()) for v in np.nonzero(solid)]
    exp = (int(0.25 / (4.0 / 40)), int(0.25 / (3.0 / 30)), int(0.25 / (2.0 / 20)))
    print("     centroid (z,y,x) = (%d,%d,%d), expected ~(%d,%d,%d) : %s"
          % (zi, yi, xi, exp[0], exp[1], exp[2],
             abs(zi - exp[0]) <= 1 and abs(yi - exp[1]) <= 1 and abs(xi - exp[2]) <= 1))
    ok &= abs(zi - exp[0]) <= 1 and abs(yi - exp[1]) <= 1 and abs(xi - exp[2]) <= 1

    # 3. Periodic stamping: a sphere on the corner must appear in all 8 octants.
    print("")
    print("periodic wrap")
    solid = voxelize(np.array([[0.0, 0.0, 0.0]]), np.array([0.3]),
                     np.array([2.0, 2.0, 2.0]), (40, 40, 40))
    octs = [solid[:20, :20, :20].sum(), solid[:20, :20, 20:].sum(),
            solid[:20, 20:, :20].sum(), solid[:20, 20:, 20:].sum(),
            solid[20:, :20, :20].sum(), solid[20:, :20, 20:].sum(),
            solid[20:, 20:, :20].sum(), solid[20:, 20:, 20:].sum()]
    equal = len(set(int(o) for o in octs)) == 1 and octs[0] > 0
    print("     corner sphere octant counts %s : %s" % (octs, equal))
    ok &= bool(equal)

    # 4. Periodic connected-component merge: a slab of pore wrapping in z is one
    #    cluster periodically and two clusters non-periodically.
    print("")
    print("periodic labelling")
    s = np.zeros((20, 10, 10), dtype=bool)
    s[5:15] = True                                   # solid slab in the middle
    c_per = pore_connectivity(s, periodic=(True, True, True))
    c_non = pore_connectivity(s, periodic=(False, False, False))
    good = c_per["n_clusters"] == 1 and c_non["n_clusters"] == 2
    print("     wrapped pore slab: periodic -> %d cluster(s), non-periodic -> %d : %s"
          % (c_per["n_clusters"], c_non["n_clusters"], good))
    ok &= bool(good)

    # The periodic flags are given (x, y, z) but the arrays are [z, y, x]:
    # an asymmetric case catches the ordering if it is ever transposed.
    # `s` above is split by a solid slab normal to z, so only z-periodicity
    # can rejoin it.
    z_only = pore_connectivity(s, periodic=(False, False, True))["n_clusters"]
    x_only = pore_connectivity(s, periodic=(True, False, False))["n_clusters"]
    good = (z_only == 1 and x_only == 2)
    print("     flag order (x,y,z): z-periodic -> %d, x-periodic -> %d "
          "(expect 1 and 2) : %s" % (z_only, x_only, good))
    ok &= bool(good)

    # 5. Inscribed-sphere pore size on a slot of exactly known width: a slab of
    #    pore 10 voxels thick admits a ball of diameter 10 and nothing larger,
    #    so the fraction below D must step from 0 to 1 between D=10 and D=11.
    print("")
    print("inscribed-sphere pore size")
    s = np.ones((30, 24, 24), dtype=bool)
    s[10:20] = False                                  # 10-voxel-thick pore slot
    h = np.array([1.0, 1.0, 1.0])
    fr = pore_size_fractions(s, h, [6, 10, 11, 14], periodic=(True, True, True))
    good = (fr[6] < 1e-6 and fr[10] < 1e-6 and fr[11] > 0.99 and fr[14] > 0.99)
    print("     10-voxel slot, frac below D for D=6,10,11,14: "
          "%.3f %.3f %.3f %.3f : %s"
          % (fr[6], fr[10], fr[11], fr[14], good))
    ok &= bool(good)

    print("-" * 72)
    print("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pack", nargs="?",
                   help="directory from yade_rcp.py, or a CSV of x,y,z,r")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--res", type=float, default=20.0,
                   help="lattice units per grain DIAMETER (default: 20)")
    p.add_argument("--cell", nargs=3, type=float, default=None,
                   help="periodic cell size, if there is no pack.json")
    p.add_argument("--shrink", type=float, default=0.0,
                   help="reduce every radius by this fraction, to open the "
                        "contacts of a close packing (e.g. 0.03)")
    p.add_argument("--pad", type=float, default=None,
                   help="empty layers, in grain diameters, at both ends of z "
                        "(default: take from pack.json; 0 keeps it periodic)")
    p.add_argument("--interface-width", type=float, default=5.0,
                   help="interface_width from params.cfg, in lu (default: 5)")
    p.add_argument("--interface-factor", type=float, default=3.0,
                   help="a throat is called unresolved below this many interface "
                        "widths (default: 3)")
    p.add_argument("--connectivity", type=int, default=2, choices=(1, 2),
                   help="1 = faces only; 2 = faces+edges, matching D3Q19 (default)")
    p.add_argument("--bytes-per-site", type=float, default=1700.0,
                   help="per-fluid-site memory of the fused GPU path (default: 1700)")
    p.add_argument("--gpu-mem", type=float, default=24.0,
                   help="GPU memory in GB, for the fit warning (default: 24)")
    p.add_argument("--compression", default=None,
                   help="TIFF compression, e.g. 'lzw' (default: none). "
                        "Must remain scanline-readable -- do not use tiles.")
    p.add_argument("--skip-edt", action="store_true",
                   help="skip the wall-distance transform (memory-hungry on big grids)")
    p.add_argument("--skip-pore-size", action="store_true",
                   help="skip the inscribed-sphere pore size (2 EDTs per ladder "
                        "entry; the slowest diagnostic on large grids)")
    p.add_argument("--skip-connectivity", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="verify the rasteriser against analytic packings and exit")

    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.pack:
        p.error("give a pack directory/CSV, or --self-test")

    centres, radii, meta = load_pack(args.pack)

    cell = args.cell or meta.get("cell")
    if cell is None:
        p.error("no cell size: pass --cell LX LY LZ or provide pack.json")
    cell = np.array(cell, dtype=float)

    if args.shrink:
        radii = radii * (1.0 - args.shrink)

    n = np.maximum(1, np.round(cell * args.res).astype(int))
    h = cell / n
    if np.max(np.abs(h / h[0] - 1.0)) > 1e-9:
        print("NOTE: cell*res is not integral on every axis, so the voxel is "
              "slightly anisotropic: h = %s d" % np.array2string(h, precision=6))

    outdir = args.out or (os.path.basename(os.path.normpath(args.pack)) +
                          "_d%g" % args.res)
    os.makedirs(outdir, exist_ok=True)

    print("rasterising %d spheres onto %d x %d x %d ..." % (len(centres), *n))
    solid = voxelize(centres, radii, cell, n)

    pad_d = args.pad if args.pad is not None else meta.get("pad", 0.0) or 0.0
    args.pad = pad_d                       # resolved value, for the metadata
    n_pad = int(round(pad_d * args.res))
    periodic = (True, True, True)
    if n_pad > 0:
        solid = pad_axis(solid, n_pad, axis_zyx=0)     # z is axis 0 of [z,y,x]
        periodic = (True, True, False)
        cell = np.array([cell[0], cell[1], cell[2] + 2 * pad_d])
        print("padded %d empty layers at both ends of z (%.2f d)" % (n_pad, pad_d))

    meta_full = dict(meta)
    if "porosity" in meta and n_pad == 0 and not args.shrink:
        meta_full["porosity"] = meta["porosity"]
    else:
        meta_full.pop("porosity", None)

    diag = report(solid, cell, h, args, meta_full, periodic)
    write_outputs(solid, outdir, cell, h, args, meta, periodic, diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
