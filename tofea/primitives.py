import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix
import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl

import numpy as np

from .solvers import SuperLU

from time import time

solver = SuperLU(diag_pivot_thresh=0.1,
                 permc_spec="MMD_AT_PLUS_A",
                 options={"SymmetricMode": True})


def _solve_coo(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    solver.factor(a)
    return solver.solve(rhs)


@jax.custom_vjp
def solve_coo(entries, indices, rhs):
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(rhs.shape), dtype=entries.dtype
    )
    return jax.pure_callback(_solve_coo, result_shape_dtype, entries, indices, rhs)


def _solve_coo_fwd(entries, indices, rhs):
    x = solve_coo(entries, indices, rhs)
    return x, (x, entries, indices, rhs)


def _solve_coo_bwd(res, g):
    ans, entries, indices, rhs = res
    x = solve_coo(entries, indices, g)
    i, j = indices
    return -x[i] * ans[j], None, x


solve_coo.defvjp(_solve_coo_fwd, _solve_coo_bwd)


def _solve_cupy(entries, indices, rhs):
    a = coo_matrix((entries, indices)).tocsc()
    x = cl.cg(cs.csr_matrix(a), cp.array(rhs))[0]
    return jnp.array(x)


@jax.custom_vjp
def solve_cupy(entries, indices, rhs):
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(rhs.shape), dtype=entries.dtype
    )
    return jax.pure_callback(_solve_cupy, result_shape_dtype, entries, indices, rhs)


def _solve_cupy_fwd(entries, indices, rhs):
    x = solve_cupy(entries, indices, rhs)
    return x, (x, entries, indices, rhs)


def _solve_cupy_bwd(res, g):
    ans, entries, indices, rhs = res
    x = solve_cupy(entries, indices, g)
    i, j = indices
    return -x[i] * ans[j], None, x


solve_cupy.defvjp(_solve_cupy_fwd, _solve_cupy_bwd)


def _cpu_mult(x, element_stiffness, elz, nelx, nely, dofmap, ix, iy, i, j, vals):
    for elx in range(nelx):
        for ely in range(nely):
            t0 = time()
            e2sdofmap = dofmap + (
                    ely + elx * (nely + 1) + elz * (nelx + 1) * (nely + 1)
            )
            v = np.multiply(x[elx, ely, elz], element_stiffness)
            ix.append(e2sdofmap[i])
            iy.append(e2sdofmap[j])
            vals = np.append(vals, (v[i, j]))
    res = np.asarray(vals).flatten()
    print("hi")
    return res


def cpu_mult(x, element_stiffness, elz, nelx, nely, dofmap, ix, iy, i, j, vals):
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(x[:, :, 0].size * element_stiffness.size * elz), dtype=x.dtype
    )
    return jax.pure_callback(_cpu_mult, result_shape_dtype, x, element_stiffness, elz, nelx, nely, dofmap, ix, iy, i, j,
                             vals)
