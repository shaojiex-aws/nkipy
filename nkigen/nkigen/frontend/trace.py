"""
Trace decorator for converting Python functions with NumPy operations to MLIR.
"""

import contextlib
import functools
import inspect
from typing import Callable, Optional

import numpy as np
from mlir import ir

from .builder import IRBuilder
from . import builder
from .traced_array import TracedArray
from .op_vtable import _to_handle, _from_handle
from .custom_op import _get_registry, _clear_registry
from .use_region import _clear_use_registry, extract_use_regions


def _normalize_shape(shape):
    if isinstance(shape, TracedArray):
        return tuple(shape.shape)
    if isinstance(shape, (list, tuple)):
        return tuple(int(d) for d in shape)
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


@contextlib.contextmanager
def _numpy_constructor_patch(source_file: str):
    """Patch NumPy constructors during tracing so they emit MLIR and return TracedArray."""
    originals = {}

    def _patch(name, fn):
        originals[name] = getattr(np, name)
        setattr(np, name, fn)

    def _make_loc():
        return ir.Location.file(source_file, 0, 0, context=ir.Context.current)

    def ones(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.full(shp, 1.0, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def zeros(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.zeros(shp, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def full(shape, fill_value, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.full(shp, fill_value, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def empty(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.empty(shp, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def array(obj, dtype=None, **kw):
        if isinstance(obj, TracedArray):
            return obj
        return originals["array"](obj, dtype=dtype, **kw)

    def asarray(obj, dtype=None, **kw):
        return array(obj, dtype=dtype, **kw)

    try:
        for name, fn in (
            ("ones", ones),
            ("zeros", zeros),
            ("empty", empty),
            ("full", full),
            ("array", array),
            ("asarray", asarray),
        ):
            _patch(name, fn)
        yield
    finally:
        for k, v in originals.items():
            setattr(np, k, v)


def trace(
    func_to_trace: Optional[Callable] = None,
    *,
    input_specs: Optional[list] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to trace a Python function with NumPy APIs into memref-native MLIR."""

    def decorator(f: Callable) -> Callable:
        func_name = name or f.__name__
        source_file = inspect.getsourcefile(f) or "unknown"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        def to_mlir(specs: Optional[list] = None, debug: bool = False,
                    *, target: str = "trn2", db_dir: Optional[str] = None):
            """Generate MLIR module from the traced function.

            ``target``/``db_dir`` are only consumed by ``knob().use(agent)``
            sites (region tiling + the agent's persistent workspace); a program
            with no agent sites ignores them.
            """
            nonlocal input_specs
            specs = specs or input_specs
            if not specs:
                raise ValueError(
                    "input_specs must be provided either in decorator or to_mlir()"
                )

            _clear_registry()
            _clear_use_registry()

            b = IRBuilder(source_file=source_file)
            arg_shapes = [s for s, _ in specs]
            arg_dtypes = [d for _, d in specs]
            handles = b.begin_function(func_name, arg_shapes, arg_dtypes)

            traced_args = [
                TracedArray(h._value, h.shape, h._elem_ty, source_file=source_file)
                for h in handles
            ]

            try:
                with _numpy_constructor_patch(source_file):
                    result = f(*traced_args)

                if isinstance(result, tuple):
                    results = list(result)
                elif isinstance(result, TracedArray):
                    results = [result]
                else:
                    raise TypeError(
                        f"Result must be a TracedArray or tuple of TracedArrays, got {type(result)}"
                    )

                for i, r in enumerate(results):
                    if not isinstance(r, TracedArray):
                        raise TypeError(
                            f"Result element {i} must be a TracedArray, got {type(r)}"
                        )

                result_handles = [_to_handle(r) for r in results]
                b.finish_function(result_handles)

                # Splice any knob().use() regions into func.call + custom op
                # BEFORE stashing declarations: extraction registers the derived
                # CustomOps that emit_custom_op_declarations then drains.
                extract_use_regions(b.module, b._func_op, target=target,
                                    db_dir=db_dir)

                custom_ops = _get_registry()
                b.emit_custom_op_declarations(custom_ops)

                # Returns canonicalized IR as text (not a Module); callers
                # stringify it anyway.  See docs/2026-06-05-nkipy-block-no-terminator-error.md.
                return b.run_canonicalize()
            finally:
                _clear_registry()
                _clear_use_registry()
                b.cleanup()

        def to_nisa(target: str = "trn2", *, dump_dir: Optional[str] = None,
                    db: Optional[str] = None) -> str:
            """Run the full knob pipeline. Returns NISA MLIR assembly.

            ``db`` is the tuning-DB dir passed to ``knob().use(agent)`` sites
            (agent workspaces); ``None`` gives agents ephemeral workspaces.
            """
            from ..driver.pipeline import apply_complete_knob_pipeline
            mlir_text = to_mlir(target=target, db_dir=db)
            return apply_complete_knob_pipeline(
                str(mlir_text), target=target, dump_dir=dump_dir,
            )

        def to_nki(target: str = "trn2", *, api_version: str = "v1",
                   dump_dir: Optional[str] = None,
                   comments: bool = False) -> str:
            """Generate kernel_builder Python source for this kernel."""
            from ..codegen.kernelbuilder import trace_to_kernelbuilder
            return trace_to_kernelbuilder(
                wrapper, target=target, api_version=api_version,
                dump_dir=dump_dir, comments=comments,
            )

        def tune(db: str, *, target: str = "trn2") -> str:
            """Offline step: give each ``knob().use(agent)`` site a persistent
            workspace under ``db/<key>/`` and run its agent there.

            The agent reads the region's kernel_builder source
            (``db/<key>/region.py``) and may dump memory/scratch/artifacts into
            its folder; nkigen records the source it returns
            (``db/<key>/kernel.py``). Returns the compiled NISA (agents are run
            as part of tracing, so this compiles the tuned program).

            POC scope: one pass over the sites with a persistent workspace. The
            full offline loop (candidate scoring, on-hardware profiling, and
            ``to_nisa(db=...)`` reading winners back so compile never searches)
            is future work — see docs/2026-06-09-nki-autotune-backend-plan.md.
            """
            import os
            os.makedirs(db, exist_ok=True)
            return to_nisa(target=target, db=db)

        wrapper.to_mlir = to_mlir
        wrapper.to_nisa = to_nisa
        wrapper.to_nki = to_nki
        wrapper.tune = tune
        wrapper.__traced__ = True
        wrapper.input_specs = input_specs

        return wrapper

    if func_to_trace is None:
        return decorator
    else:
        return decorator(func_to_trace)
