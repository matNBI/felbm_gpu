# Sphere-pack geometry pipeline

Builds the 3D granular domains used in *"Dynamic multiphase flow triggers chaotic
mixing in porous media"* and feeds them to felbm through the existing
`domain_geometry = 3d_image` path. **No C++ changes are needed.**

```
yade_rcp.py  ──▶  spheres.csv + pack.json  ──▶  voxelize_spheres.py  ──▶  image/pack.tif
   (Yade DEM)        (analytic, resolution-free)      (rasterise)          + domain.cfg
```

Keeping the packing in its analytic form (centres + radii, with the grain
diameter as the length unit) is the point of splitting it in two: one DEM run
serves every lattice resolution, and a resolution study is just a second call to
the voxelizer.

---

## 1. `yade_rcp.py` — random close packing

Reproduces the paper's recipe (Methods, *Porous Domains*), which uses Yade
[Angelidakis et al., *Comput. Phys. Commun.* **304**, 109293 (2024)]:

> place N spheres in a triple-periodic domain, numerically shrink it with fixed
> aspect ratios until jammed; depending on the difference between the jammed box
> and the target, increase or decrease N and repeat, until the jammed-state box
> size is within 0.1% of the target.

```bash
yade yade_rcp.py -- --box 20 20 30 --out pack_20x20x30 --seed 1
```

Compression is *strain*-controlled and isotropic (`O.cell.velGrad ∝ I`), which is
what preserves the aspect ratio exactly. Contacts are frictionless, which is the
standard route to the RCP branch (φ ≈ 0.64) rather than a looser frictional
random packing. The outer loop uses the fact that the jammed solid fraction is
essentially independent of N, so the next guess is `N = φ_measured · V_target /
v_sphere`; with N ~ 10³–10⁴ a single-sphere change moves the box by 0.003–0.03%,
which is why the paper's 0.1% tolerance is reachable at all.

Two deliberate departures from the paper:

- **No inlet/outlet padding by default.** The paper padded 0.5 d at each end to
  impose uniform inlet/outlet velocities for a *drainage* run. The intended use
  here is triple-periodic body-force-driven co-flow in a statistically steady
  state, where padding would break the periodicity. Use `--pad 0.5` to reproduce
  the paper's 4×4×8 d³ box.
- **A much larger target box**, because a steady co-flow has to contain the
  largest fluid clusters. In 2D the paper needed 60d × 90d for Ca ≳ 10⁻³.

Change `--seed` for independent realisations. Output is `spheres.csv` (x,y,z,r,
with d = 1) and `pack.json` (cell, N, φ, coordination number, convergence
history).

**No Yade?** Any generator works — the voxelizer only needs a CSV of `x,y,z,r`
plus a cell size (`--cell LX LY LZ`). But Yade is what the paper used, and there
is a stronger option still: ref. 82 of the paper deposits *"Meshes and analysis
scripts"*. If the original sphere list is in that dataset, voxelizing **it**
turns the LBM-vs-FEM comparison into a same-geometry comparison rather than a
statistically-equivalent one.

---

## 2. `voxelize_spheres.py` — rasterise and diagnose

```bash
./voxelize_spheres.py pack_20x20x30 --res 20 --out geom_d20
./voxelize_spheres.py pack_20x20x30 --res 32 --shrink 0.03 --out geom_d32_s3
./voxelize_spheres.py --self-test
```

`--res` is lattice units per grain **diameter**, i.e. d in lu — the number you
actually reason about.

### Output format

Written to match `DomainInitializer_3DImage` exactly:

| | |
|---|---|
| one TIFF **page** per z-slice | the reader iterates `TIFFReadDirectory` |
| row = y, column = x | so `id = col + size_x*(row + size_y*z)` ↔ voxel (x,y,z) |
| 8-bit, 1 sample/pixel, **strips not tiles** | the reader uses `TIFFReadScanline`, which cannot read tiled TIFFs |
| grain = 255, pore = 0 | read with `interface_solid = true`, `invert_image = false` |

A ready-to-use `domain.cfg` is emitted alongside, with every key the initializer
reads (it calls `get_value` unconditionally, so none may be missing), plus
`geometry.json` with all the diagnostics for the record.

In `settings.cfg`:

```
domain_geometry = 3d_image
domain_cfg_file = domain.cfg
image_periodic  = true      # genuine triple-periodic wrap, no padding layers
use_open_bnd    = false     # body-force drive
empty_layers    = 0
```

`size_x/size_y/size_z` are overridden by the image dimensions.

### Diagnostics, and why they are there

**Porosity** is checked against the analytic value from the sphere list, so the
rasterisation error is visible rather than assumed.

**Pore connectivity** is labelled with the periodic faces merged (including the
diagonal links, and only wrapping in-plane axes that are themselves periodic).
The default `--connectivity 2` (face + edge, 18 of 26) is what **D3Q19** actually
sees; `--connectivity 1` is the stricter face-only test. Watch the *isolated /
dead pore* fraction: pore voxels the solver can never reach.

**Inscribed-sphere pore size** is the one that decides whether a packing is
usable at all for a *two-phase* run. A voxel has local pore diameter ≥ D if some
ball of radius D/2 fits entirely in the pore space and covers it — the standard
maximal-inscribed-sphere definition, computed by a morphological opening.

> Note it reports this *and* the wall distance, and they are not the same thing.
> The wall distance (plain EDT) is dominated by near-wall voxels even inside a
> wide pore, so it badly overstates how tight a medium is. Only the
> inscribed-sphere number should be read as a pore or constriction width.

The reason this matters: **a diffuse-interface method cannot represent an
interface in a constriction narrower than a few interface widths.** With
`interface_width = 5` lu and RCP's constrictions at ~0.3 d, that is a hard lower
bound on d in lattice units, and it collides with the memory limit from the other
direction. The reported line

```
pore volume with local diameter < 15.0 lu (=3 W) : 0.78
```

is the number to drive down — by raising `--res`, lowering `interface_width`, or
`--shrink`ing the grains. The paper does the analogous thing in 2D: *"The initial
diameter is set to d_init = 1.2 d … we then shrink the cylinders to a diameter d,
which produces … good connectivity of the pore space."* Whatever you choose,
report it, and settle it with a single-phase permeability check against the FEM
before running any two-phase case.

**GPU memory** is estimated from the fluid-site count at 1160 B/site, against
`--gpu-mem`. This is the constraint that caps the box, and hence the largest
cluster the domain can hold, and hence the lowest Ca you can honestly report.

That 1160 is measured rather than estimated — `felbm_gpu` reported 10129.8 MiB in
use after init for 9,168,806 fluid sites on the fused + `stream_inplace` path in
single precision. It replaces an earlier figure of 1700, which overstated memory
by 47% and made resolutions look infeasible that are not: at 1160 a 8×8×12 d³ RCP
pack at `d = 40` is 20.7 GiB and fits a 24 GB card. Raise `--bytes-per-site` for
configurations that allocate more — without `stream_inplace` the h2/g2 ping-pong
buffers add ~150 B/site at D3Q19 single, and a double build roughly doubles the
distribution arrays.

### Self-test

`--self-test` verifies, with no external data:

- solid fraction converges to the exact value for simple-cubic, BCC and FCC
  packings (π/6, √3π/8, π/√18) over a 4× refinement — these have touching
  spheres, so they are also the worst case for voxelised contacts;
- the TIFF round-trip preserves (pages, rows, cols) = (nz, ny, nx) and a sphere
  placed at a known fraction of the cell lands at the right voxel;
- a sphere on the cell corner appears in all eight octants (periodic stamping);
- periodic labelling joins a pore slab that wraps, and the periodic flags are in
  (x, y, z) order while the arrays are [z, y, x];
- the inscribed-sphere size on a slot of exactly 10 voxels steps from 0 to 1
  between D = 10 and D = 11.

Run it after any edit. It takes about a minute.

### Options worth knowing

| flag | |
|---|---|
| `--shrink F` | reduce every radius by fraction F, to open the contacts of a close packing |
| `--pad D` | D diameters of empty space at both ends of z, the paper's inlet/outlet geometry. Breaks z-periodicity; the emitted `domain.cfg` says so and gives the open-boundary keys |
| `--interface-width` / `--interface-factor` | `interface_width` from `params.cfg` (default 5 lu) and how many widths count as resolved (default 3) |
| `--skip-pore-size`, `--skip-edt` | the distance transforms are the slow part; skip them on very large grids |
| `--cell LX LY LZ` | when feeding a bare CSV with no `pack.json` |

---

## Suggested first use

Before committing to a production box, run the voxelizer over a resolution and
shrink ladder on one small packing and read off the trade-off:

```bash
for res in 12 16 20 24 32; do
  ./voxelize_spheres.py pack_8x8x12 --res $res --out scan_d$res
done
```

`d` climbs the pore-size curve and the memory curve at the same time, and where
those two cross is what sets the feasible Ca range for the 3D sweep.
