from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING
from abc import abstractmethod, ABCMeta


from types_abc import ImmediateResult, NumericalABC, IntegerABC

if TYPE_CHECKING:
    from scope import Scope
    from types_info import (
        BaseType,
        PrimitiveType,
    )
    from lexer import TokenKind


class GenResult:
    @staticmethod
    def singular(item: ImmediateResult, gen_backend: GenBackend):
        res = GenResult(gen_backend)
        res.insert(0, [item])
        return res

    def __init__(self, backend: GenBackend) -> None:
        self.backend = backend
        self._results: list[ImmediateResult] = []

    def __getitem__(self, index: int) -> ImmediateResult:
        return self._results[index]

    def __len__(self):
        return len(self._results)

    def results(self):
        return self._results.copy()

    def last(self):
        return self._results[-1]

    def append_ir(self, result: ImmediateResult):
        self.insert(len(self), [result])

    def append(self, var: str, ir: str):
        self.append_ir(ImmediateResult(var, ir))

    def append_str(self, ir: str):
        self.append(None, ir)

    def insert(self, index: int, results: list[ImmediateResult]):
        for i, res in enumerate(results):
            self._results.insert(index + i, res)

    def concat(self, other: GenResult):
        self.insert(len(self), other.results())

    def pop(self, index: int):
        return self._results.pop(index)


class BlockIR(NamedTuple):
    result: GenResult
    start_label: str
    end_label: str


class IfArm(NamedTuple):
    cond: GenResult
    body: BlockIR


class GenBackend(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def literal(scope: Scope, typ: PrimitiveType, value: str) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: GenResult,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def address(
        scope: Scope,
        var: str,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def dereference(
        scope: Scope,
        typ: PrimitiveType,
        expr: GenResult,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def var_assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def deref_assignment(
        ref: GenResult, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def cast(
        scope: Scope, expr: GenResult, from_: NumericalABC, to: NumericalABC
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def logical_connective(
        scope: Scope,
        op: TokenKind,
        left: GenResult,
        right: GenResult,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def binary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        left: GenResult,
        right: GenResult,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def if_expr(scope: Scope, arms: list[IfArm]) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def while_expr(scope: Scope, cond: GenResult, body: BlockIR) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def var(scope: Scope, name: str, typ: PrimitiveType) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def func_call(
        scope: Scope,
        name: str,
        params: list[tuple[GenResult, BaseType]],
        typ: PrimitiveType,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def putchar(expr: GenResult, typ: IntegerABC) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def return_stmt(expr: GenResult) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def var_declare(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def block(
        scope: Scope,
        start_label: str,
        end_label: str,
        statements: list[GenResult],
        gen_labels: bool,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def function(
        scope: Scope,
        name: str,
        arguments: list[tuple[str, BaseType]],
        body: BlockIR,
        return_type: BaseType,
    ) -> GenResult: ...

    @staticmethod
    @abstractmethod
    def program(scope: Scope, functions: list[GenResult]) -> GenResult: ...
