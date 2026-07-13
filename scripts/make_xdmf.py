#!/usr/bin/env python3
"""
Build an XDMF file for ParaView from felbm_gpu HDF5 output.

Two modes:

  Fields (default):
    geometry.h5      : coords (N x 3 int, grid position of each fluid site), size (3)
    <prefix>_<it>.h5 : density, concentration, u_x, u_y, u_z, pressure (each length N)
    -> point cloud at the fixed fluid-site coords, colored by the fields; velocity is
       assembled into a vector from u_x/u_y/u_z. Needs geometry.h5 (output_xdmf=true).

  Particles (--particles):
    particles_<it>.h5 : position (M x 3), velocity (M x 3), id (M)
    -> moving point cloud (each snapshot uses its own `position` as geometry), with
       velocity as a vector attribute and id as a scalar. No geometry.h5 needed.
       Requires HDF5 particle output (particles_format = h5).

The field arrays are 1-D fluid-site lists with no built-in geometry, so ParaView
can't place them on its own; this wraps them (and the particle dumps) in a temporal
collection so the whole run loads as one time series. Needs h5py.

Usage:
  python make_xdmf.py [DIR=.] [--prefix P] [--geom geometry.h5] [--dt 1.0] [--particles]
Then open  DIR/<prefix>.xdmf  in ParaView.
"""
import os, sys, glob, re, argparse

try:
    import h5py
except ImportError:
    sys.exit("make_xdmf.py needs h5py  (pip install h5py)")


def iter_of(path, prefix):
    m = re.search(re.escape(prefix) + r"_(\d+)\.h5$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_files(directory, prefix):
    files = [p for p in glob.glob(os.path.join(directory, "%s_*.h5" % prefix))
             if iter_of(p, prefix) >= 0]
    files.sort(key=lambda p: iter_of(p, prefix))
    return files


def di(dim, src, ntype="Float", prec=8):
    p = ('Precision="%d" ' % prec) if ntype == "Float" else ''
    return ('<DataItem Dimensions="%s" NumberType="%s" %sFormat="HDF">%s</DataItem>'
            % (dim, ntype, p, src))


def float_precision(h5file, names):
    """itemsize (4 or 8) of the first present dataset, so the XDMF matches
    float32/float64 output; the datasets may be gzip-compressed (transparent here)."""
    with h5py.File(h5file, "r") as f:
        for k in names:
            if k in f:
                return int(f[k].dtype.itemsize)
    return 8


def build_fields(a):
    geom_path = os.path.join(a.dir, a.geom)
    if not os.path.exists(geom_path):
        sys.exit("geometry file not found: %s\n"
                 "  (run felbm_gpu with output_xdmf = true to produce it)" % geom_path)
    with h5py.File(geom_path, "r") as f:
        N = int(f["coords"].shape[0])
    files = find_files(a.dir, a.prefix)
    if not files:
        sys.exit("no %s_*.h5 files found in %s" % (a.prefix, a.dir))
    with h5py.File(files[0], "r") as f:
        keys = set(f.keys())
    scalars = [k for k in ("density", "concentration", "pressure") if k in keys]
    has_vel = all(k in keys for k in ("u_x", "u_y", "u_z"))
    prec = float_precision(files[0], ("density", "concentration", "pressure", "u_x"))
    gname = os.path.basename(a.geom)

    L = []
    for p in files:
        it = iter_of(p, a.prefix); fn = os.path.basename(p)
        L += ['   <Grid Name="t%d" GridType="Uniform">' % it,
              '    <Time Value="%g"/>' % (it * a.dt),
              '    <Topology TopologyType="Polyvertex" NumberOfElements="%d"/>' % N,
              '    <Geometry GeometryType="XYZ">',
              '     ' + di("%d 3" % N, "%s:/coords" % gname, "Int"),
              '    </Geometry>']
        for s in scalars:
            L += ['    <Attribute Name="%s" AttributeType="Scalar" Center="Node">' % s,
                  '     ' + di("%d" % N, "%s:/%s" % (fn, s), prec=prec),
                  '    </Attribute>']
        if has_vel and a.vector_velocity:
            # velocity as an XDMF Vector via a Function JOIN. Convenient (glyphs /
            # streamlines) but ParaView's Xdmf3 reader can CRASH on Function items
            # when stepping through time — opt-in only.
            L += ['    <Attribute Name="velocity" AttributeType="Vector" Center="Node">',
                  '     <DataItem ItemType="Function" Function="JOIN($0, $1, $2)" Dimensions="%d 3">' % N]
            for c in ("u_x", "u_y", "u_z"):
                L += ['      ' + di("%d" % N, "%s:/%s" % (fn, c), prec=prec)]
            L += ['     </DataItem>', '    </Attribute>']
        elif has_vel:
            # default: three plain scalar components (no Function -> crash-proof).
            # Recombine in ParaView with the "Merge Vector Components" filter if needed.
            for c in ("u_x", "u_y", "u_z"):
                L += ['    <Attribute Name="%s" AttributeType="Scalar" Center="Node">' % c,
                      '     ' + di("%d" % N, "%s:/%s" % (fn, c), prec=prec),
                      '    </Attribute>']
        L += ['   </Grid>']
    return L, len(files), "fields: %s%s" % (", ".join(scalars), ", velocity" if has_vel else "")


def build_particles(a):
    files = find_files(a.dir, a.prefix)
    if not files:
        sys.exit("no %s_*.h5 files found in %s\n"
                 "  (particle HDF5 output needs particles_format = h5)" % (a.prefix, a.dir))
    with h5py.File(files[0], "r") as f:
        keys = set(f.keys())
    if "position" not in keys:
        sys.exit("%s has no 'position' dataset — is this HDF5 particle output?" % os.path.basename(files[0]))
    has_vel = "velocity" in keys
    has_id = "id" in keys
    prec = float_precision(files[0], ("position", "velocity"))

    L = []
    for p in files:
        it = iter_of(p, a.prefix); fn = os.path.basename(p)
        with h5py.File(p, "r") as f:
            M = int(f["position"].shape[0])
        L += ['   <Grid Name="t%d" GridType="Uniform">' % it,
              '    <Time Value="%g"/>' % (it * a.dt),
              '    <Topology TopologyType="Polyvertex" NumberOfElements="%d"/>' % M,
              '    <Geometry GeometryType="XYZ">',
              '     ' + di("%d 3" % M, "%s:/position" % fn, prec=prec),
              '    </Geometry>']
        if has_vel:
            L += ['    <Attribute Name="velocity" AttributeType="Vector" Center="Node">',
                  '     ' + di("%d 3" % M, "%s:/velocity" % fn, prec=prec),
                  '    </Attribute>']
        if has_id:
            L += ['    <Attribute Name="id" AttributeType="Scalar" Center="Node">',
                  '     ' + di("%d" % M, "%s:/id" % fn, "Int"),
                  '    </Attribute>']
        L += ['   </Grid>']
    tail = "particles" + (", velocity" if has_vel else "") + (", id" if has_id else "")
    return L, len(files), tail


def main():
    ap = argparse.ArgumentParser(description="XDMF writer for felbm_gpu HDF5 output")
    ap.add_argument("dir", nargs="?", default=".", help="folder with the .h5 files")
    ap.add_argument("--prefix", default=None, help="file prefix (default: output, or particles with --particles)")
    ap.add_argument("--geom", default="geometry.h5", help="geometry file for field mode (default: geometry.h5)")
    ap.add_argument("--dt", type=float, default=1.0, help="LBM steps per snapshot index (Time value)")
    ap.add_argument("--particles", action="store_true", help="build XDMF for particle dumps instead of fields")
    ap.add_argument("--vector-velocity", dest="vector_velocity", action="store_true",
                    help="write field velocity as a JOIN'd vector (convenient, but can crash ParaView's "
                         "Xdmf3 reader on time-step; default writes u_x/u_y/u_z as scalars)")
    a = ap.parse_args()
    if a.prefix is None:
        a.prefix = "particles" if a.particles else "output"

    body, nsteps, summary = (build_particles(a) if a.particles else build_fields(a))

    L = ['<?xml version="1.0" ?>', '<Xdmf Version="2.0">', ' <Domain>',
         '  <Grid Name="felbm" GridType="Collection" CollectionType="Temporal">']
    L += body
    L += ['  </Grid>', ' </Domain>', '</Xdmf>']

    out = os.path.join(a.dir, "%s.xdmf" % a.prefix)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s   (%d snapshots, %s)" % (out, nsteps, summary))


if __name__ == "__main__":
    main()
