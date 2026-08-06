#!/usr/bin/env python3
"""
periodic_disk_mesh.py -- doubly-periodic 2D disk pack, meshed for Twoasis AND
exported as centres so felbm_gpu can build the identical geometry.

    ./periodic_disk_mesh.py --Lx 4 --Ly 8 --rad 0.5 --N 19 --res 0.05 \
                            --out ~/code/twoasis/meshes

Writes, using Twoasis's own naming convention (Porous2D.get_fname):

    periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.h5    dolfin mesh
    periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.dat   "x y r" per disk
    ...                                                    .centres.txt (same, for us)

WHY THIS EXISTS
---------------
Twoasis ships no meshes, and `meshes/pore_mesh.py` emits spheres (3D only). But
needing to write one is an opportunity rather than a chore: if the SAME disk
centres drive both codes, the comparison is exact rather than statistical, and
any difference in sd(t) is solver behaviour and nothing else.

The domain is centred on the origin, [-Lx/2, Lx/2] x [-Ly/2, Ly/2], because
Porous2D's PBC maps +Lx/2 -> -Lx/2. The .dat is "x y r" per row: Porous2D's
mark_subdomains reads obst[:, :2] as centres and obst[:, 2] as radius.

PERIODICITY IS THE WHOLE DIFFICULTY
-----------------------------------
dolfin's PeriodicBoundary needs vertices on opposite edges to MATCH, so the mesh
itself has to be periodic, not merely the geometry. Two things follow:

  * a disk within `rad` of an edge is added again, translated by +/-Lx or +/-Ly,
    so the solid is continuous across the wrap. Corner disks need up to four
    copies. Without this the pack is not periodic and the flow sees a solid-free
    channel along the boundary, which would dominate everything.
  * every boundary curve gets an explicit gmsh `setPeriodic` constraint. The cut
    disks split each edge into several curves, so left and right are paired by
    matching bounding boxes under the translation, not by tag order.

RSA placement uses the PERIODIC distance, so disks may straddle an edge.
"""
import argparse
import os
import sys

import numpy as np


def place_disks(Lx, Ly, rad, N, gap, seed, max_tries=200000):
    """RSA in a doubly-periodic box; returns (N,2) centres. Periodic distance,
    so disks are free to straddle an edge -- that is what makes the pack
    homogeneous rather than having a depleted strip at the boundary."""
    rng = np.random.default_rng(seed)
    L = np.array([Lx, Ly])
    pos = []
    tries = 0
    dmin = 2 * rad + gap
    while len(pos) < N and tries < max_tries:
        tries += 1
        x = rng.random(2) * L - L / 2
        if pos:
            d = np.asarray(pos) - x
            d -= L * np.round(d / L)              # minimum image
            if np.min(np.linalg.norm(d, axis=1)) < dmin:
                continue
        pos.append(x)
    if len(pos) < N:
        sys.exit(f"could only place {len(pos)}/{N} disks in {tries} tries -- "
                 f"lower --N, --rad or --gap")
    return np.asarray(pos)


def images(centres, Lx, Ly, rad):
    """Disk centres plus periodic copies for any disk reaching past an edge."""
    out = []
    for cx, cy in centres:
        sx = [0.0]
        sy = [0.0]
        if cx - rad < -Lx / 2: sx.append(+Lx)
        if cx + rad > +Lx / 2: sx.append(-Lx)
        if cy - rad < -Ly / 2: sy.append(+Ly)
        if cy + rad > +Ly / 2: sy.append(-Ly)
        for dx in sx:
            for dy in sy:
                out.append((cx + dx, cy + dy))
    return out


def build_mesh(centres, Lx, Ly, rad, res, msh_path, verbose=False):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("periodic_porous")

    rect = gmsh.model.occ.addRectangle(-Lx / 2, -Ly / 2, 0, Lx, Ly)
    tools = [(2, gmsh.model.occ.addDisk(cx, cy, 0, rad, rad))
             for cx, cy in images(centres, Lx, Ly, rad)]
    gmsh.model.occ.cut([(2, rect)], tools)
    gmsh.model.occ.synchronize()

    # Pair boundary curves across each period by bounding box, then constrain.
    eps = 1e-6 * min(Lx, Ly)

    def curves_on(axis, value):
        lo = [-Lx / 2 - eps, -Ly / 2 - eps, -eps]
        hi = [+Lx / 2 + eps, +Ly / 2 + eps, +eps]
        lo[axis] = value - eps
        hi[axis] = value + eps
        return gmsh.model.getEntitiesInBoundingBox(*lo, *hi, 1)

    for axis, L in ((0, Lx), (1, Ly)):
        lows = curves_on(axis, -L / 2)
        highs = curves_on(axis, +L / 2)
        if len(lows) != len(highs):
            gmsh.finalize()
            sys.exit(f"axis {axis}: {len(lows)} curves on the low edge but "
                     f"{len(highs)} on the high edge -- the pack is not periodic")
        affine = np.eye(4)
        affine[axis, 3] = L
        for dim, ltag in lows:
            lbb = np.array(gmsh.model.getBoundingBox(1, ltag))
            match = None
            for _, htag in highs:
                hbb = np.array(gmsh.model.getBoundingBox(1, htag))
                shifted = hbb.copy()
                shifted[axis] -= L
                shifted[axis + 3] -= L
                if np.allclose(lbb, shifted, atol=1e-6 * max(Lx, Ly)):
                    match = htag
                    break
            if match is None:
                gmsh.finalize()
                sys.exit(f"axis {axis}: no periodic partner for curve {ltag}")
            gmsh.model.mesh.setPeriodic(1, [match], [ltag], affine.flatten().tolist())

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)   # meshio-friendly
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return n_nodes


def to_dolfin_h5(msh_path, h5_path):
    """gmsh .msh -> dolfin Mesh -> HDF5 with dataset "mesh", which is what
    Porous2D's mesh() reads. Built through MeshEditor rather than XDMF: fewer
    moving parts, and no dependence on XDMF encoding quirks in dolfin 2019.1."""
    import meshio
    from dolfin import Mesh, MeshEditor, HDF5File, MPI

    m = meshio.read(msh_path)
    pts = m.points[:, :2]
    tri = None
    for cb in m.cells:
        if cb.type == "triangle":
            tri = cb.data
    if tri is None:
        sys.exit("no triangles in the generated mesh")

    mesh = Mesh()
    ed = MeshEditor()
    ed.open(mesh, "triangle", 2, 2)
    ed.init_vertices(len(pts))
    ed.init_cells(len(tri))
    for i, p in enumerate(pts):
        ed.add_vertex(i, p)
    for i, c in enumerate(tri):
        ed.add_cell(i, c.astype(np.uintp))
    ed.close()

    with HDF5File(MPI.comm_world, h5_path, "w") as h5:
        h5.write(mesh, "mesh")
    return mesh.num_vertices(), mesh.num_cells()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Lx", type=float, default=4.0)
    p.add_argument("--Ly", type=float, default=8.0)
    p.add_argument("--rad", type=float, default=0.5)
    p.add_argument("--R", type=float, default=0.55, help="only names the file (Twoasis convention)")
    p.add_argument("--N", type=int, default=19)
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--gap", type=float, default=0.06, help="min surface-to-surface separation")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--centres", default=None, help="read centres from this file instead of placing")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    if a.centres:
        centres = np.loadtxt(a.centres)[:, :2]
        a.N = len(centres)
        print(f"using {a.N} centres from {a.centres}")
    else:
        centres = place_disks(a.Lx, a.Ly, a.rad, a.N, a.gap, a.seed)

    phi = 1.0 - a.N * np.pi * a.rad ** 2 / (a.Lx * a.Ly)
    stem = ("periodic_porous_Lx{Lx:g}_Ly{Ly:g}_r{rad:g}_R{R:g}_N{N}_dx{res:g}"
            .format(Lx=a.Lx, Ly=a.Ly, rad=a.rad, R=a.R, N=a.N, res=a.res))
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)

    print(f"{a.N} disks r={a.rad} in {a.Lx} x {a.Ly}  ->  porosity {phi:.4f}")

    # The .dat MUST carry the periodic image centres too, not just the primaries.
    # Porous2D's Walls.inside() marks a boundary point as wall if it lies within
    # `r` of some centre in this file, and that is what applies no-slip. A disk
    # straddling an edge has its wrapped half near the OPPOSITE edge, far from
    # its own centre, so with primaries alone those facets go unmarked, no-slip
    # is silently missing on part of the solid, and the velocity diverges within
    # a few steps. (Symptom: phim leaves [-1,1] and E_kin blows up regardless of
    # dt -- reducing dt makes it worse, not better, which is the tell that it is
    # a boundary-condition bug and not CFL.)
    dat = np.array(images(centres, a.Lx, a.Ly, a.rad))
    np.savetxt(base + ".dat",
               np.column_stack([dat, np.full(len(dat), a.rad)]),
               header="x y r  (includes periodic images -- see note in source)")
    # felbm applies periodicity itself, so it wants the primaries only.
    np.savetxt(base + ".centres.txt", centres, header="x y  (primary centres only)")
    print(f"  .dat has {len(dat)} disks ({len(centres)} primary + "
          f"{len(dat)-len(centres)} periodic images)")

    n_nodes = build_mesh(centres, a.Lx, a.Ly, a.rad, a.res, base + ".msh", a.verbose)
    nv, nc = to_dolfin_h5(base + ".msh", base + ".h5")
    print(f"  mesh: {nv} vertices, {nc} triangles  (gmsh reported {n_nodes} nodes)")
    print(f"  wrote {base}.h5 / .dat / .centres.txt")


if __name__ == "__main__":
    main()
