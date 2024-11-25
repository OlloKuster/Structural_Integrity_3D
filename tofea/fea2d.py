from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from time import time

import jax.numpy as jnp
from jax import Array

from tofea.elements import Q4Element_T
from tofea.primitives import solve_coo
from .primitives import solve_cupy


@dataclass
class FEA2D(ABC):
    fixed: Array
    dx: float = 0.5
    dy: float = 0.5

    @property
    @abstractmethod
    def dof_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def element(self) -> Array:
        ...

    @property
    @abstractmethod
    def dofmap(self) -> Array:
        ...

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny = self.fixed.shape[:2]
        return nx - 1, ny - 1

    @cached_property
    def dofs(self) -> Array:
        return jnp.arange(self.fixed.size, dtype=jnp.uint32)

    @cached_property
    def fixdofs(self) -> Array:
        return self.dofs[self.fixed.ravel()]

    @cached_property
    def freedofs(self) -> Array:
        return self.dofs[~self.fixed.ravel()]

    @cached_property
    def index_map(self) -> Array:
        indices = jnp.concatenate([self.freedofs, self.fixdofs])
        imap = jnp.zeros_like(self.dofs)
        imap = imap.at[indices].set(self.dofs)
        return imap

    @cached_property
    def e2sdofmap(self) -> Array:
        nx, ny = self.shape
        x, y = jnp.unravel_index(jnp.arange(nx * ny), (nx, ny))
        idxs = self.dof_dim * (y + x * (ny + 1))
        return jnp.add(self.dofmap[None], idxs[:, None].astype(jnp.uint32))

    @cached_property
    def keep_indices(
        self,
    ) -> tuple[Array, Array]:
        r = jnp.arange(self.dofmap.size)
        i, j = jnp.meshgrid(r, r)
        ix = self.e2sdofmap[:, i].ravel()
        iy = self.e2sdofmap[:, j].ravel()
        keep = jnp.isin(ix, self.freedofs) & jnp.isin(iy, self.freedofs)
        indices = jnp.stack([self.index_map[ix][keep], self.index_map[iy][keep]])
        return keep, indices

    def global_mat(self, x: Array) -> tuple[Array, Array]:
        x = jnp.reshape(x, (-1, 1, 1)) * self.element[None]
        x = x.ravel()
        keep, indices = self.keep_indices
        return x[keep], indices

    def solve(self, x: Array, b: Array) -> Array:
        print("assemble matrix...", end=" ")
        t0 = time()
        data, indices = self.global_mat(x)
        print(f"done: {time() - t0:.3f}s")
        print("gpu solve...", end=" ")
        t0 = time()
        u_nz = solve_cupy(data, indices, b.ravel()[self.freedofs])
        print(f"done: {time() - t0:.3f}s")
        z = jnp.zeros(self.fixdofs.size)
        u = jnp.concatenate([u_nz, z])[self.index_map]
        return u



@dataclass
class FEA2D_T(FEA2D):
    dof_dim: int = 1
    k: float = 1.0

    @cached_property
    def element(self) -> Array:
        return Q4Element_T(k=self.k, dx=self.dx, dy=self.dy).element

    @cached_property
    def dofmap(self) -> Array:
        _, nely = self.shape
        return jnp.r_[1, (nely + 2), (nely + 1), 0].astype(jnp.uint32)

    def temperature(self, x: Array, b: Array) -> Array:
        return self.solve(x, b)
