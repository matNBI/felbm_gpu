#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yade_rcp.py -- triple-periodic random close packing (RCP) of identical spheres.

Reproduces the packing recipe of Linga et al., "Dynamic multiphase flow triggers
chaotic mixing in porous media" (Methods, "Porous Domains"), which uses Yade
[Angelidakis et al., Comput. Phys. Commun. 304, 109293 (2024)]:

    "we specify a target box size 4d x 4d x 7d and place N spheres in a
     triple-periodic domain, which we then numerically shrink with fixed aspect
     ratios, using the discrete element code Yade, until the system is jammed.
     Depending on the difference between the box size in the jammed state and the
     target size, we increase or decrease N and repeat the compression process,
     until the jammed-state box size is within 0.1% of the target size."

Differences from the paper, on purpose:

  * No inlet/outlet padding by default.  The paper padded 0.5d at top and bottom
    to impose uniform inlet/outlet velocities for a drainage run.  The intended
    use here is a *triple-periodic, body-force-driven co-flow* in a statistically
    steady state, so padding would break the periodicity we want.  ``--pad``
    is available if you do want to reproduce the paper's 4x4x8 d^3 box.
  * The default target box is much larger than 4x4x7 d^3, because a statistically
    steady co-flow has to contain the largest fluid clusters.

Compression is *strain*-controlled and isotropic (``O.cell.velGrad`` proportional
to the identity), so the cell aspect ratio is preserved exactly -- that is what
"shrink with fixed aspect ratios" means.  Contacts are frictionless, which is the
standard way to reach the RCP branch (phi ~ 0.64) rather than a looser
frictional random packing.

Usage
-----
    yade yade_rcp.py -- --box 20 20 30 --out pack_20x20x30

    yade yade_rcp.py -- --box 4 4 7 --pad 0.5 --out pack_paper   # paper geometry

Output (written to ``--out``):
    spheres.csv    x,y,z,r   (one sphere per line, length unit = d = 1)
    pack.json      cell size, N, phi_solid, coordination number, provenance

Feed both to ``voxelize_spheres.py`` to produce the TIFF stack that
``domain_geometry = 3d_image`` reads.

Note on units: the sphere *diameter* is the length unit (d = 1), so all box
sizes on the command line are in grain diameters and the emitted radii are 0.5.
"""

from __future__ import print_function

import json
import math
import os
import sys

# --------------------------------------------------------------------------- #
#  Argument parsing.  Yade passes everything after "--" through in sys.argv.
# --------------------------------------------------------------------------- #

def parse_args(argv):
    import argparse

    p = argparse.ArgumentParser(
        prog="yade yade_rcp.py --",
        description="Triple-periodic RCP of identical spheres (Yade).")

    p.add_argument("--box", nargs=3, type=float, default=[20.0, 20.0, 30.0],
                   metavar=("LX", "LY", "LZ"),
                   help="target box size in grain diameters (default: 20 20 30). "
                        "The aspect ratio is preserved exactly during compression; "
                        "only the overall scale is set by jamming.")
    p.add_argument("--out", default="pack",
                   help="output directory (default: pack)")
    p.add_argument("--seed", type=int, default=1,
                   help="RNG seed for the initial cloud (default: 1). "
                        "Change this to get independent packing realisations.")
    p.add_argument("--phi-target", type=float, default=0.64,
                   help="expected jammed solid fraction, used only for the first "
                        "guess at N (default: 0.64)")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="relative box-size tolerance for the outer loop on N "
                        "(default: 1e-3, i.e. the paper's 0.1%%)")
    p.add_argument("--max-outer", type=int, default=12,
                   help="maximum outer iterations on N (default: 12)")
    p.add_argument("--phi-init", type=float, default=0.30,
                   help="solid fraction of the initial dilute cloud (default: 0.30). "
                        "Lower is safer for makeCloud but slower to compress.")
    p.add_argument("--strain-rate", type=float, default=2e-4,
                   help="isotropic engineering strain rate per unit time during "
                        "compression (default: 2e-4). Lower = slower but safer.")
    p.add_argument("--p-jam", type=float, default=1e-5,
                   help="jamming pressure as a fraction of the Young modulus "
                        "(default: 1e-5). Sets the residual contact overlap: "
                        "delta/R ~ (p/E)^(2/3), so 1e-5 gives ~5e-4 R.")
    p.add_argument("--unbalanced", type=float, default=2e-3,
                   help="unbalanced-force threshold for static equilibrium "
                        "(default: 2e-3)")
    p.add_argument("--pad", type=float, default=0.0,
                   help="pad this many diameters of empty space at both ends of "
                        "the LAST axis, as the paper does for inlet/outlet "
                        "(default: 0.0 = keep the cell triple-periodic). "
                        "Padding is recorded in pack.json and applied by the "
                        "voxelizer, not by the DEM.")
    p.add_argument("--max-steps", type=int, default=4000000,
                   help="hard cap on DEM steps per compression (default: 4e6)")

    # Yade puts the script name first; everything after a bare "--" is ours.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
#  One compression run: N spheres of radius R, target aspect ratio -> jammed cell
# --------------------------------------------------------------------------- #

def compress_to_jamming(N, R, aspect, args):
    """Place N spheres in a dilute periodic cell of the given aspect ratio and
    compress isotropically until jammed.  Returns (cell_size, phi, coord_z)."""

    from yade import pack, utils
    from yade.wrapper import (ForceResetter, InsertionSortCollider, Bo1_Sphere_Aabb,
                              InteractionLoop, Ig2_Sphere_Sphere_ScGeom,
                              Ip2_FrictMat_FrictMat_FrictPhys,
                              Law2_ScGeom_FrictPhys_CundallStrack,
                              NewtonIntegrator, PyRunner, FrictMat)
    from yade import O
    from minieigen import Matrix3, Vector3

    young = 1e8

    # --- initial dilute cell, same aspect ratio as the target ---------------
    v_sph = N * (4.0 / 3.0) * math.pi * R ** 3
    v_cell0 = v_sph / args.phi_init
    a = [float(x) for x in aspect]
    scale0 = (v_cell0 / (a[0] * a[1] * a[2])) ** (1.0 / 3.0)
    L0 = [scale0 * ai for ai in a]

    O.reset()
    O.periodic = True
    O.cell.hSize = Matrix3(L0[0], 0, 0, 0, L0[1], 0, 0, 0, L0[2])

    mat = O.materials.append(
        FrictMat(young=young, poisson=0.3, frictionAngle=0.0, density=2600.0))

    sp = pack.SpherePack()
    n_placed = sp.makeCloud((0, 0, 0), (L0[0], L0[1], L0[2]),
                            rMean=R, rRelFuzz=0.0, num=N,
                            periodic=True, seed=args.seed)
    if n_placed < N:
        raise RuntimeError(
            "makeCloud placed only %d of %d spheres at phi_init=%.3f; "
            "lower --phi-init" % (n_placed, N, args.phi_init))

    for c, r in sp:
        O.bodies.append(utils.sphere(c, r, material=mat))

    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=False),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()],
        ),
        NewtonIntegrator(damping=0.4),
        PyRunner(command="check_state()", iterPeriod=200, label="checker"),
    ]
    O.dt = 0.4 * utils.PWaveTimeStep()

    # Isotropic engineering strain rate -> the cell shrinks at fixed aspect ratio.
    rate = args.strain_rate
    O.cell.velGrad = Matrix3(-rate, 0, 0, 0, -rate, 0, 0, 0, -rate)

    state = {"phase": "compress", "done": False}
    p_jam = args.p_jam * young

    def check_state():
        s = utils.getStress()
        p = -(s[0][0] + s[1][1] + s[2][2]) / 3.0

        if state["phase"] == "compress":
            if p > p_jam:
                # Freeze the cell and let the packing relax into equilibrium at
                # constant volume.  Jamming is declared when the residual force
                # imbalance is small *and* the pressure has not collapsed.
                O.cell.velGrad = Matrix3.Zero
                state["phase"] = "relax"
        else:
            if utils.unbalancedForce() < args.unbalanced:
                if p > 0.1 * p_jam:
                    state["done"] = True
                    O.pause()
                else:
                    # Relaxed away the contact stress: not jammed yet, resume.
                    O.cell.velGrad = Matrix3(-rate, 0, 0, 0, -rate, 0, 0, 0, -rate)
                    state["phase"] = "compress"

    # PyRunner resolves `command` in the yade namespace, so publish it there.
    import __main__
    __main__.check_state = check_state

    O.run(args.max_steps, True)
    if not state["done"]:
        raise RuntimeError("compression did not reach a jammed state within "
                           "%d steps (phase=%s)" % (args.max_steps, state["phase"]))

    L = [O.cell.size[0], O.cell.size[1], O.cell.size[2]]
    phi = v_sph / (L[0] * L[1] * L[2])

    n_contacts = sum(1 for i in O.interactions if i.isReal)
    coord = 2.0 * n_contacts / float(N)

    centres = [(b.state.pos[0], b.state.pos[1], b.state.pos[2]) for b in O.bodies]
    return L, phi, coord, centres


# --------------------------------------------------------------------------- #

def main():
    args = parse_args(sys.argv)

    R = 0.5                       # d = 1 is the length unit
    v_sph1 = (4.0 / 3.0) * math.pi * R ** 3
    target = [float(x) for x in args.box]
    v_target = target[0] * target[1] * target[2]

    # First guess: N such that the jammed cell has the target volume at phi_target.
    N = int(round(args.phi_target * v_target / v_sph1))

    print("target box   : %.4f x %.4f x %.4f  (d = 1)" % tuple(target))
    print("first guess  : N = %d  (phi_target = %.3f)" % (N, args.phi_target))
    print("")

    history = []
    best = None

    for it in range(args.max_outer):
        L, phi, coord, centres = compress_to_jamming(N, R, target, args)

        # Isotropic compression preserves the aspect ratio, so a single scalar
        # measures the mismatch.
        scale = (v_target / (L[0] * L[1] * L[2])) ** (1.0 / 3.0)
        rel = abs(scale - 1.0)

        print("iter %2d : N = %6d  L = %.4f %.4f %.4f  phi = %.5f  Z = %.2f  "
              "rel.mismatch = %.3e" % (it, N, L[0], L[1], L[2], phi, coord, rel))

        history.append({"iter": it, "N": N, "L": L, "phi": phi,
                        "coordination": coord, "rel_mismatch": rel})

        if best is None or rel < best["rel"]:
            best = {"rel": rel, "N": N, "L": L, "phi": phi,
                    "coord": coord, "centres": centres}

        if rel < args.tol:
            break

        # phi at jamming is essentially independent of N, so the N that fills the
        # target volume at the *measured* phi is the natural next guess.
        N_new = int(round(phi * v_target / v_sph1))
        if N_new == N:
            N_new = N + (1 if scale > 1.0 else -1)
        N = N_new
    else:
        print("\nWARNING: did not reach the %.1f%% tolerance in %d iterations; "
              "using the best of them (rel = %.3e)."
              % (100 * args.tol, args.max_outer, best["rel"]))

    # --- rescale the best packing so the cell is *exactly* the target ---------
    # The correction is below --tol by construction, i.e. sub-0.1% on the radius,
    # which is far smaller than one voxel at any resolution we would run.
    L = best["L"]
    sx, sy, sz = target[0] / L[0], target[1] / L[1], target[2] / L[2]
    # Isotropic compression means sx == sy == sz to round-off; take the geometric
    # mean for the radius so the sphere stays a sphere.
    R_out = R * (sx * sy * sz) ** (1.0 / 3.0)
    centres = [((c[0] * sx) % target[0],
                (c[1] * sy) % target[1],
                (c[2] * sz) % target[2]) for c in best["centres"]]

    phi_out = len(centres) * (4.0 / 3.0) * math.pi * R_out ** 3 / v_target

    outdir = args.out
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, "spheres.csv"), "w") as f:
        f.write("# x,y,z,r  (length unit: grain diameter d = 1)\n")
        for c in centres:
            f.write("%.10f,%.10f,%.10f,%.10f\n" % (c[0], c[1], c[2], R_out))

    meta = {
        "generator": "yade_rcp.py",
        "reference": ("Linga et al., Methods/'Porous Domains'; "
                      "Yade: Angelidakis et al., CPC 304, 109293 (2024)"),
        "packing": "RCP, monodisperse, frictionless, triple-periodic",
        "length_unit": "grain diameter d = 1",
        "cell": target,
        "N": len(centres),
        "radius": R_out,
        "phi_solid": phi_out,
        "porosity": 1.0 - phi_out,
        "coordination_number": best["coord"],
        "seed": args.seed,
        "pad": args.pad,
        "pad_axis": 2,
        "tol": args.tol,
        "history": history,
    }
    with open(os.path.join(outdir, "pack.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("")
    print("wrote %s/spheres.csv  (N = %d, r = %.6f d)"
          % (outdir, len(centres), R_out))
    print("      %s/pack.json" % outdir)
    print("      phi_solid = %.5f   porosity = %.5f   Z = %.2f"
          % (phi_out, 1.0 - phi_out, best["coord"]))
    if args.pad > 0:
        print("      NOTE: pad = %.2f d recorded for axis z; the cell is no longer "
              "periodic along z once the voxelizer applies it." % args.pad)


if __name__ == "__main__":
    main()
