from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy
from jax import Array


@dataclass(frozen=True, slots=True, kw_only=True)
class Element:
    dx: float = 0.5
    dy: float = 0.5
    dz: float = 0.5
    eps: float = 1e-6
    dtype: type = np.float64

    @staticmethod
    def _b_entries(
        rule: Iterable[int],
        shape_funcs: Iterable[sympy.Expr],
        clist: Iterable[sympy.Symbol],
    ) -> tuple[Any, ...]:
        shape_list = np.concatenate([x * np.asarray(rule) for x in shape_funcs])
        return tuple(map(sympy.diff, shape_list, clist))


class Q4Element(Element):
    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return sympy.symbols("a b x y", real=True)

    @property
    def shape_funcs(self) -> list[sympy.Expr]:
        a, b, x, y = self.symbols
        return [
            (a - x) * (b - y) / (4 * a * b),
            (a + x) * (b - y) / (4 * a * b),
            (a + x) * (b + y) / (4 * a * b),
            (a - x) * (b + y) / (4 * a * b),
        ]


@dataclass(frozen=True, slots=True)
class Q4Element_T(Q4Element):
    k: float = 1.0

    @cached_property
    def element(self) -> Array:
        a, b, x, y = self.symbols
        k = sympy.symbols("k", real=True)

        B = sympy.Matrix(
            [
                self._b_entries([1], self.shape_funcs, 4 * [x]),
                self._b_entries([1], self.shape_funcs, 4 * [y]),
            ]
        )

        C = sympy.Matrix([[k, 0], [0, k]])

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b))
        K = np.array(K.subs({a: self.dx, b: self.dy, k: self.k}), dtype=self.dtype)
        K[np.abs(K) < self.eps] = 0
        return jnp.asarray(K)


class H8Element(Element):
    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return sympy.symbols("a b c x y z", real=True)

    @property
    def shape_funcs(self) -> list[sympy.Expr]:
        a, b, c, x, y, z = self.symbols
        return [
            (a - x) * (b - y) * (c - z) / (8 * a * b * c),
            (a + x) * (b - y) * (c - z) / (8 * a * b * c),
            (a + x) * (b + y) * (c - z) / (8 * a * b * c),
            (a - x) * (b + y) * (c - z) / (8 * a * b * c),
            (a - x) * (b - y) * (c + z) / (8 * a * b * c),
            (a + x) * (b - y) * (c + z) / (8 * a * b * c),
            (a + x) * (b + y) * (c + z) / (8 * a * b * c),
            (a - x) * (b + y) * (c + z) / (8 * a * b * c),
        ]



@dataclass(frozen=True, slots=True)
class H8Element_T(H8Element):
    k: float = 1.0

    @cached_property
    def element(self) -> Array:
        a, b, c, x, y, z = self.symbols
        k = sympy.symbols("k", real="True")

        B = sympy.Matrix(
            [
                self._b_entries([1], self.shape_funcs, 8 * [x]),
                self._b_entries([1], self.shape_funcs, 8 * [y]),
                self._b_entries([1], self.shape_funcs, 8 * [z]),
            ]
        )

        C = sympy.Matrix([[k, 0, 0], [0, k, 0], [0, 0, k]])

        dK = B.T * C * B
        K = dK.integrate((x, -a, a), (y, -b, b), (z, -c, c))
        K = np.array(
            K.subs({a: self.dx, b: self.dy, c: self.dz, k: self.k}),
            dtype=self.dtype,
        )
        K[np.abs(K) < self.eps] = 0
        return jnp.asarray(K)