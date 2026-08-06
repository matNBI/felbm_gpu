"""Alias: twoasis targets the post-2022 FEniCS split, where legacy UFL was renamed
`ufl_legacy`. conda-forge's fenics 2019.1.0 ships that same codebase under its
original name `ufl`, and dolfin 2019.1.0 is built against it -- so aliasing is
correct here, not a workaround. Installing a real ufl-legacy wheel would instead
give twoasis a DIFFERENT UFL class hierarchy from the one dolfin uses.

SUBMODULES MUST BE ALIASED TOO. `from ufl_legacy.tensors import ListTensor` would
otherwise re-execute ufl/tensors.py as a fresh module, re-running the @ufl_type
decorators and tripping
    assert Expr._ufl_num_typecodes_ == len(Expr._ufl_all_handler_names_)
because every UFL type ends up registered twice. Mapping each already-imported
ufl.* onto ufl_legacy.* makes the two names resolve to one module object.
"""
import sys
import ufl as _ufl

for _name, _mod in list(sys.modules.items()):
    if _name == "ufl" or _name.startswith("ufl."):
        sys.modules["ufl_legacy" + _name[3:]] = _mod

if not hasattr(_ufl, "ListTensor"):
    from ufl.tensors import ListTensor as _ListTensor
    _ufl.ListTensor = _ListTensor

sys.modules[__name__] = _ufl
