"""Affine map construction (createStandardNisaMap mirror) and the
operand-kwargs builders handed to the nisa.<op> Python builders."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

from .access import _Access

def _create_standard_nisa_map(
    ctx: nk_ir.Context,
    num_iter_dims: int,
    num_symbols: int,
    num_results: int,
    dropped_dims: list[bool],
) -> nk_ir.AffineMap:
    """Python mirror of C++ ``createStandardNisaMap``.

    d0 -> first kept position, d(N-1) -> last kept position, middle iter dims
    map to middle kept positions in order. Dropped dims get pure symbol or
    constant expressions; any remaining result has symbol or 0.
    """
    kept_positions = [
        i for i in range(num_results)
        if not (dropped_dims and i < len(dropped_dims) and dropped_dims[i])
    ]

    position_to_dim: dict[int, int] = {}
    if kept_positions and num_iter_dims > 0:
        position_to_dim[kept_positions[0]] = 0
        if num_iter_dims > 1 and len(kept_positions) > 1:
            position_to_dim[kept_positions[-1]] = num_iter_dims - 1
            mid_dim = 1
            for k in range(1, len(kept_positions) - 1):
                if mid_dim + 1 >= num_iter_dims:
                    break
                position_to_dim[kept_positions[k]] = mid_dim
                mid_dim += 1

    exprs: list[nk_ir.AffineExpr] = []
    for i in range(num_results):
        is_dropped = bool(dropped_dims) and i < len(dropped_dims) and dropped_dims[i]
        if is_dropped:
            if i < num_symbols:
                exprs.append(nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(nk_ir.AffineConstantExpr.get(0))
        elif i in position_to_dim:
            dim_expr = nk_ir.AffineDimExpr.get(position_to_dim[i])
            if i < num_symbols:
                exprs.append(dim_expr + nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(dim_expr)
        else:
            if i < num_symbols:
                exprs.append(nk_ir.AffineSymbolExpr.get(i))
            else:
                exprs.append(nk_ir.AffineConstantExpr.get(0))
    return nk_ir.AffineMap.get(num_iter_dims, num_symbols, exprs)


def _build_nisa_map(
    ctx: nk_ir.Context, num_iter_dims: int, access: _Access
) -> nk_ir.Attribute:
    amap = _create_standard_nisa_map(
        ctx,
        num_iter_dims,
        num_symbols=len(access.indices),
        num_results=access.base_rank,
        dropped_dims=access.dropped_dims,
    )
    return nisa.flatten_affine_map(amap, ctx)


def _operand_kwargs(
    prefix: str,
    access: _Access,
    flat_map: nk_ir.Attribute,
    tile_shape: list[int],
    tile_par_dims: int = 1,
) -> dict:
    return {
        f"{prefix}_memloc": access.base,
        f"{prefix}_indices": access.indices,
        f"{prefix}_ap": flat_map,
        f"{prefix}_static_tile_shape": list(tile_shape),
        f"{prefix}_tile_par_dims": tile_par_dims,
    }


def _empty_operand_kwargs(prefix: str) -> dict:
    """Kwargs for an 'omitted' optional operand (ap=None, memloc=None)."""
    return {
        f"{prefix}_memloc": None,
        f"{prefix}_indices": [],
        f"{prefix}_ap": None,
        f"{prefix}_static_tile_shape": [],
        f"{prefix}_tile_par_dims": 0,
    }


def _scalar_operand_kwargs(prefix: str, scalar: nk_ir.Value) -> dict:
    return {
        f"{prefix}_memloc": scalar,
        f"{prefix}_indices": [],
        f"{prefix}_ap": None,
        f"{prefix}_static_tile_shape": [],
        f"{prefix}_tile_par_dims": 0,
    }
