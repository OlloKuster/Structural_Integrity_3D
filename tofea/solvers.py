from abc import ABC, abstractmethod
from functools import partial

from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class Solver(ABC):
    @abstractmethod
    def factor(self, m: csc_matrix) -> None:
        ...

    @abstractmethod
    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class SuperLU(Solver):
    _ctx: dict = {}

    def __init__(self, **options):
        self.options = options
        self._ctx["splu"] = partial(splu, **options)

    def factor(self, m: csc_matrix) -> None:
        self._ctx["factorization"] = self._ctx["splu"](m)

    def solve(self, rhs: NDArray, transpose: bool = False) -> NDArray:
        return self._ctx["factorization"].solve(rhs, trans="T" if transpose else "N")

    def clear(self) -> None:
        self._ctx.pop("factorization", None)


def get_solver(solver: str) -> Solver:
    match solver:
        case "SuperLU":
            return SuperLU(
                diag_pivot_thresh=0.1,
                permc_spec="MMD_AT_PLUS_A",
                options={"SymmetricMode": True},
            )
        case _:
            raise ValueError(f"Invalid solver: {solver}")
