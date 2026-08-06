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

