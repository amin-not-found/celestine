from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Optional

from lexer import TokenKind
from scope import Scope


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


UnaryOperator = Callable[["PrimitiveType", str, Scope], ImmediateResult]
BinaryOperator = Callable[
    ["PrimitiveType", ImmediateResult, ImmediateResult, Scope],
    ImmediateResult,
]


class PrimitiveType:
    __metaclass__ = ABCMeta

    unary_operators: dict[TokenKind, UnaryOperator] = {}
    binary_operators: dict[TokenKind, BinaryOperator] = {}

    @classmethod
    def unary_op_func(cls, op: TokenKind) -> UnaryOperator:
        return getattr(cls, cls.unary_operators[op])

    @classmethod
    def binary_op_func(cls, op: TokenKind) -> BinaryOperator:
        return getattr(cls, cls.binary_operators[op])

    @classmethod
    @abstractmethod
    def assign(cls, name: str, a: ImmediateResult, scope: Scope) -> ImmediateResult: ...


class NumericalType(PrimitiveType):
    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def negative(cls, a: ImmediateResult, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def add(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def subtract(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def multiply(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def divide(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def gt(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def lt(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ge(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def le(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def eq(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ne(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    unary_operators: dict = {
        **PrimitiveType.unary_operators,
        TokenKind.MINUS: "negative",
    }
    binary_operators = {
        **PrimitiveType.binary_operators,
        TokenKind.PLUS: "add",
        TokenKind.MINUS: "subtract",
        TokenKind.ASTERISK: "multiply",
        TokenKind.SLASH: "divide",
        TokenKind.GT: "gt",
        TokenKind.LT: "lt",
        TokenKind.GE: "ge",
        TokenKind.LE: "le",
        TokenKind.EQ: "eq",
        TokenKind.NE: "ne",
    }


class Integer(NumericalType):
    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def logical_not(cls, a: ImmediateResult, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def logical_or(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def logical_and(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def reminder(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_and(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_or(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_xor(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def left_shift(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def right_shift(
        cls, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ) -> ImmediateResult: ...

    unary_operators: dict = {
        **NumericalType.unary_operators,
        TokenKind.BANG: "logical_not",
    }

    binary_operators = {
        **NumericalType.binary_operators,
        TokenKind.L_AND: "logical_or",
        TokenKind.L_OR: "logical_and",
        TokenKind.PERCENT: "reminder",
        TokenKind.AMPERSAND: "bit_and",
        TokenKind.V_BAR: "bit_or",
        TokenKind.CARET: "bit_xor",
        TokenKind.SHIFT_L: "left_shift",
        TokenKind.SHIFT_R: "right_shift",
    }


class I32(Integer):
    # pylint: disable=abstract-method
    __metaclass__ = ABCMeta
    size = 4


class I64(Integer):
    # pylint: disable=abstract-method
    __metaclass__ = ABCMeta
    size = 8
