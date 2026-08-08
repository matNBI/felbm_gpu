# Running Twoasis (the paper's own solver) locally

**2026-08-06.** Working, in ~30 minutes, without Docker. This is the reference
implementation for Linga et al. — `twoasis/problems/TPfracStep/Porous2D.py` has
exactly Table S2's parameters, so anything it produces is ground truth for the
comparison rather than another of our inferences from the PDF.

## What did NOT work

- **`quay.io/fenicsproject/stable:*`** — every tag is from 2019–2020 and uses the
  schema-v1 manifest, which containerd 2.1+ refuses outright:
  `media type application/vnd.docker.distribution.manifest.v1+prettyjws is no
  longer supported`. The prebuilt legacy FEniCS images are effectively dead.
- **Their `docker/Dockerfile`** — builds PETSc 3.13.3, SLEPc and dolfin from
  source against bitbucket clones. Hours, and the bitbucket FEniCS repos are of
  uncertain longevity. Kept as a fallback; not needed.
- **`fenics=2019.2.0.dev20240219`** — listed on anaconda.org but not built for
  linux-64. Only 2019.1.0 is.

## What works

```bash
curl -L -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda create -y -n fenics -c conda-forge --override-channels \
  fenics=2019.1.0 h5py gmsh meshio numpy scipy matplotlib
cd /home/mathies/code/twoasis && $HOME/miniforge3/envs/fenics/bin/pip install --no-deps .
```

Installs to `$HOME`, so **no root is needed** — and unlike a container it is
directly drivable from an agent shell, which matters for iteration.

Two things are required at run time:

**1. `PKG_CONFIG_PATH`.** conda-forge ships `dolfin.pc` but no activation script
that points at it, so dolfin's JIT raises
`Could not find DOLFIN pkg-config file`. Every invocation needs:

```bash
export PKG_CONFIG_PATH=$HOME/miniforge3/envs/fenics/lib/pkgconfig
```

**2. A `ufl_legacy` shim**, at
`$HOME/miniforge3/envs/fenics/lib/python3.13/site-packages/ufl_legacy.py`.

twoasis targets the post-2022 FEniCS split, where legacy UFL was renamed
`ufl_legacy`. conda's fenics 2019.1.0 ships that same codebase as plain `ufl`,
and its dolfin is built against it — so aliasing is *correct here*, not a fudge.
Installing a real `ufl-legacy` wheel would be the mistake: twoasis would then
hold a different UFL class hierarchy from the one dolfin uses.

The shim must alias **submodules too**. `from ufl_legacy.tensors import
ListTensor` otherwise re-executes `ufl/tensors.py` as a fresh module, re-running
the `@ufl_type` decorators and tripping

    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_handler_names_)

because every UFL type ends up registered twice. Mapping each already-imported
`ufl.*` onto `ufl_legacy.*` makes both names resolve to one module object (108
aliases). `ListTensor` also sits in `ufl.tensors` in 2019.1.0 but at top level in
later ufl-legacy, so it is re-exported.

Only four names are needed across twoasis: `Coefficient`, `ListTensor`,
`max_value`, `min_value`, plus one `import ufl_legacy as ufl`.

## Smoke test

```bash
cd /tmp/scratch && PKG_CONFIG_PATH=$HOME/miniforge3/envs/fenics/lib/pkgconfig \
  $HOME/miniforge3/envs/fenics/bin/twoasis TPfracStep problem=TaylorBubble2D T=0.2
```

Runs 10 timesteps in ~16 s: phase field, tentative velocity u0/u1, velocity
update. Version caveat: this is dolfin **2019.1.0**; the paper used 2019.2.0. So
far nothing has needed the difference, but it is the first thing to suspect if
results disagree in a way the physics cannot explain.

## The remaining blocker

Every porous problem reads a pre-generated mesh that is NOT in the repo:

- `Porous2D` wants `meshes/periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.h5`
  **and** a matching `.dat` of obstacle centres (`np.loadtxt`).
- `CylArr2D` wants `meshes/cyl_arr_Lx{Lx}_Ly{Ly}_rad{rad}_dx{res}.h5`.

`meshes/pore_mesh.py` generates spheres (3D only), so a 2D periodic disk mesher
has to be written. That is also the opportunity: generating the disk centres
ourselves lets the SAME centres drive both codes, making the comparison exact
rather than statistical.

`meshes/yade_rcp.py` here is the same script as our `scripts/geometry/yade_rcp.py`
— the geometry pipelines already share provenance.

## Useful facts read straight out of the source

`Porous2D.py` defaults, which are Table S2 exactly:

```python
T=100.0, Lx=4, Ly=8, rad=0.5, N=19, res=0.05, R=0.55, dt=0.05,
rho=[1, 1], mu=[1, 1], theta=np.pi/3, epsilon=0.05, sigma=5.0, M=0.0001,
F0=[0., 10.], checkerboard=[12, 18], velocity_degree=1
```

- **`checkerboard=[12, 18]` is a block COUNT, not a length.** On the paper's
  60d x 90d production domain that is exactly 5d blocks — independent
  confirmation of our checkerboard, from source rather than from prose.
- **`F0=[0, 10]`** is Table S3's driving force `|f| = 10`, i.e. Ca ~ 2.3e-2 in
  the paper's geometry — near the optimum.
- The shipped default is a 4d x 8d box with 19 disks, so 0.33d checkerboard
  blocks: the same block count as production but a much smaller domain, hence a
  different regime. Match deliberately, not by accident.


## On `camel` (80 cores) -- dolfin 2019.2.0, the paper's exact version

Reached over `ssh -p 8122 mathies@localhost`. Better than the local conda env,
which is 2019.1.0. Twoasis is already installed there under `~/.local`, but two
things get in the way and BOTH env vars are needed:

```bash
export PYTHONNOUSERSITE=1 DOLFIN_ALLOW_USER_SITE_IMPORTS=1
export PYTHONPATH=$HOME/code/twoasis:$HOME/code/ufl_compat
```

- pip-installed `fenics-ufl/ffc/dijitso` in `~/.local` shadow the apt dolfin and
  make it refuse to import (`Unknown ufl object type FiniteElement`).
  `PYTHONNOUSERSITE=1` hides them -- but it also hides twoasis, which lives in
  the same directory, hence the fresh clone at `~/code/twoasis`.
- dolfin's check fires on the mere PRESENCE of those directories, so
  `DOLFIN_ALLOW_USER_SITE_IMPORTS=1` is needed as well. Together they are safe:
  one silences the check, the other guarantees user-site is genuinely unused.

### The reverse `ufl` shim (`~/code/ufl_compat/ufl.py`)

`twoasis/solvers/TPfracStep/__init__.py` does

```python
import ufl_legacy as ufl
if not isinstance(Constant, ufl.Coefficient):
    import ufl
```

`Constant` is a CLASS, so that isinstance is always False and the fallback always
fires. Upstream only gets away with it because their Docker installs BOTH ufl
2022.1.0 and ufl-legacy. camel has only `ufl_legacy` -- which is what dolfin
2019.2.0 is built against -- so aliasing `ufl` -> `ufl_legacy` is MORE correct
than upstream: `ufl.min_value/max_value/sin/cos` are applied to dolfin Functions
a few lines later and must come from dolfin's own hierarchy. Alias submodules
too, for the same double-registration reason as the local shim.

Note this is the exact mirror of the local situation: there twoasis wanted
`ufl_legacy` and conda supplied `ufl`; here it wants `ufl` and apt supplies
`ufl_legacy`.

### Scaling, measured

20 steps on the 187k-vertex 20d x 30d mesh:

| ranks | 4 | 16 | 40 | 80 |
|---|---|---|---|---|
| time | 39.9 s | 14.1 s | **10.2 s** | 12.0 s |

Sweet spot ~40 ranks; 80 is too thin at this mesh size. A 1.7M-vertex mesh would
have ~21k vertices/rank at 80, which should scale better, but that is untested.

### WARNING: check VALUES, not completion

The 20d x 30d run at 40 ranks reached `Time = 1.0000e+00` in the scaling test and
was declared fine. It was not: launched long, it diverged inside ~700 steps
(`phim` -0.58, `E_kin` 3e43) and then burned 40 cores for hours on saturated
garbage. The scaling test had only been checked for COMPLETION.

Always assert `abs(phim) < 0.01` and `sd` of order 1 before trusting any twoasis
run, and before sizing anything from it.

Meshes are generated LOCALLY (camel has the gmsh binary but not the Python API)
with `scripts/geometry/periodic_disk_mesh.py`, then copied to
`~/code/twoasis/meshes`. The 60d x 90d production mesh builds fine: 2613 disks,
1,671,784 vertices, 3,180,104 triangles.

`phim` is a WEAK detector, incidentally. The 60x90 no-forcing run below holds
`phim` at 5e-3 while `E_kin` reaches 3.5e9 -- mass is conserved throughout a total
blow-up. Assert on `E_kin` and `E_int` as well; `E_int` frozen to 5 digits across
consecutive steps means the phase field has stopped evolving (see G below).

## The 60d x 90d case does not run -- three bugs found, none of them the cause

Unresolved as of 2026-08-08. Recorded because the bugs are real and independent
of this campaign, and because the eliminations are expensive to redo.

**What it does.** Blows up at step 3-4 from `E_kin` ~1e-2 to 1e9+, at every dt
from 0.02 down to 0.001, at 4/16/40/80 ranks, with and without body force.

**What is excluded, with data.**

| suspect | evidence against |
|---|---|
| dt | 0.01 -> 0.001 changed nothing |
| Bo / Ca / forcing | `F0=[0,0]` blows up identically, same step |
| rank count | 80 ranks are fine on 20x30 |
| mesh dimensions | exactly 60x90; density 310/unit^2 vs 311 at 20x30; porosity 0.380 vs 0.379 |
| sealed pores | 60x90 is fully connected. The 4x8 mesh, which WORKS, is the one with an isolated 3-vertex pocket |
| sliver cells | min cell area 1.0e-4 vs 2.2e-4 at 20x30, medians identical to 4 digits, no cell below q=0.01 |
| grain near-contacts | min gap 0.060 vs 0.061 at 20x30; nothing within one cell width in either |
| iteration cap | `maximum_iterations=5000` made it WORSE (`E_kin` 1.8e27 by step 4). Failure is `DIVERGED_BREAKDOWN` at 30-60 iterations, not `DIVERGED_ITS` at the cap |

The failing solve is `pressure_solve` (`IPCS.py:283`), identified by running with
`krylov_solvers={"error_on_nonconvergence":true}` and reading the traceback.

**Three bugs fixed on camel** (uncommitted, `.orig` backups beside each file;
revert with `git checkout <file>`). Each was verified against a 20x30 regression
that must reproduce `E_kin` 8.2915e-3 -> 1.9087e-2 and `E_int` 4.9692 -> 4.5414.
All three leave that case identical to every printed digit.

1. `twoasis/common/__init__.py`, `convert()` -- `input.iter()` is a Python 2
   leftover and raises `AttributeError`, so EVERY dict-valued command-line
   parameter crashed the parser before the solver started. The note-to-self at
   the top of that file documenting `velocity_update_solver='{"method":...}'`
   describes a feature that has never worked under Python 3. Behind it,
   `str.encode('utf-8')` returned `str` in Python 2 but returns BYTES in
   Python 3, which would have made the merged keys silently fail to match.
2. `Porous2D.py`, `PBC` -- `near()` defaults to `DOLFIN_EPS` = 3e-16, which is
   ABSOLUTE, while coordinate rounding error grows with magnitude (4.0e-15 at
   Lx/2 = 2, 8.9e-14 at 30). Nodes failing the match scale 6/112 at 4x8,
   18/463 at 20x30, 57/1180 at 60x90. Worse, `map()`'s last branch was an
   unguarded `else`, so a RIGHT-edge node failing `near(x[0], Lx/2)` fell
   through to the top-edge case and was mapped to `(x0, x1-Ly)` -- a WRONG
   constraint tying a boundary dof to an interior point, not a missing one.
   Fixed with `tol = 1e-9` (far above the 1e-13 noise, far below the 0.05 cell).
3. `IPCS.py`, `pressure_solve` -- the domain is fully periodic with no pressure
   Dirichlet BC, so the operator is singular with a constant null space.
   `attach_pressure_nullspace` sets `.null_space` on `as_backend_type(Apt[0])`,
   a DIFFERENT object, while the guard tests `hasattr(Apt[0], 'null_space')`.
   The guard is therefore always False -- provably, since a True guard would
   raise `AttributeError` on `p_sol.null_space`, which nothing assigns. So the
   RHS was never projected and GMRES was always handed an inconsistent singular
   system. Fixed via `Apt[2]`, the `VectorSpaceBasis` that does have
   `.orthogonalize()`.

**Effect of each, on 60x90:**

| | step-1 `E_kin` | breakdowns | blows up at |
|---|---|---|---|
| baseline | 1.25 | -- | step 3 |
| + PBC tol | 7.98e-3 | 1360 | step 4 |
| + null-space RHS | 6.19e-3 | 480 | step 4 |

Right direction, not a cure. Step-1 `E_kin` is now within 25% of the healthy
20x30 value and breakdowns are down 65%, but it still fails.

**A trap worth remembering.** Moving the velocity and phase-field solves to
`gmres`+`hypre_amg` made 60x90 run 10 steps with bounded `E_kin` and looked like
a fix. It was not: `E_int` was frozen at 1.8730 for all 10 steps, and the 20x30
control froze at 5.2712 too. AMG is wrong for the Cahn-Hilliard block, so with
`nonzero_initial_guess=True` the solve returned its input unchanged and the flow
was stable because there was no interface dynamics left. Without the small-mesh
regression this would have been recorded as success.

**Before spending more on this**, note the 60x90 run is only a domain-size
control on a result already in hand -- twoasis at 20x30 gives `sd` = 1.008
against the paper's 0.333, agreeing with felbm and disagreeing with the paper.
The same control is far cheaper in felbm, which works at every size.

