from typing import NamedTuple
from abc import abstractmethod, ABCMeta

from lexer import TokenKind
from scope import Scope
from types_info import (
    BaseType,
    PrimitiveType,
)

from types_abc import ImmediateResult, NumericalABC, IntegerABC


class GenResult:
    @staticmethod
    def singular(item: ImmediateResult, gen_backend: "GenBackend"):
        res = GenResult(gen_backend)
        res.insert(0, [item])
        return res

    def __init__(self, backend: "GenBackend") -> None:
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

    def concat(self, other: "GenResult"):
        self.insert(len(self), other.results())

    def pop(self, index: int):
        self._results.pop(index)


class BlockIR(NamedTuple):
    result: GenResult
    start_label: str
    end_label: str


class IfArm(NamedTuple):
    cond: GenResult
    body: BlockIR


def gen_method(func):
    return staticmethod(abstractmethod(func))


class GenBackend(metaclass=ABCMeta):
    # pylint: disable=unused-argument

    @gen_method
    def literal(scope: Scope, typ: PrimitiveType, value: str) -> GenResult: ...

    @gen_method
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: GenResult,
    ) -> GenResult: ...

    @gen_method
    def address(
        scope: Scope,
        var: str,
    ) -> GenResult: ...

    @gen_method
    def dereference(
        scope: Scope,
        typ: PrimitiveType,
        expr: GenResult,
    ) -> GenResult: ...

    @gen_method
    def var_assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @gen_method
    def deref_assignment(
        scope: Scope, ref: GenResult, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @gen_method
    def cast(
        scope: Scope, expr: GenResult, from_: NumericalABC, to: NumericalABC
    ) -> GenResult: ...

    @gen_method
    def logical_connective(
        scope: Scope,
        op: TokenKind,
        left: GenResult,
        right: GenResult,
        typ: PrimitiveType,
    ) -> GenResult: ...

    @gen_method
    def binary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        left: GenResult,
        right: GenResult,
    ) -> GenResult: ...

    @gen_method
    def if_expr(scope: Scope, arms: list[IfArm]) -> GenResult: ...

    @gen_method
    def while_expr(scope: Scope, cond: GenResult, body: BlockIR) -> GenResult: ...

    @gen_method
    def var(scope: Scope, name: str, typ: PrimitiveType) -> GenResult: ...

    @gen_method
    def func_call(
        scope: Scope,
        name: str,
        params: list[tuple[GenResult, BaseType]],
        typ: PrimitiveType,
    ) -> GenResult: ...

    @gen_method
    def putchar(expr: GenResult, typ: IntegerABC) -> GenResult: ...

    @gen_method
    def return_stmt(expr: GenResult) -> GenResult: ...

    @gen_method
    def var_declare(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult: ...

    @gen_method
    def block(
        scope: Scope,
        start_label: str,
        end_label: str,
        statements: list[GenResult],
        gen_labels: bool,
    ) -> GenResult: ...

    @gen_method
    def function(
        scope: Scope,
        name: str,
        arguments: list[tuple[str, BaseType]],
        body: BlockIR,
        return_type: BaseType,
    ) -> GenResult: ...

    @gen_method
    def program(scope: Scope, functions: list[GenResult]) -> GenResult: ...
