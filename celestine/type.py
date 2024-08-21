from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Optional, Type

from lexer import TokenKind
from scope import Scope, BaseType


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


UnaryOperator = Callable[[Type["PrimitiveType"], str, Scope], ImmediateResult]
BinaryOperator = Callable[
    [Type["PrimitiveType"], str, str, Scope],
    ImmediateResult,
]


class PrimitiveType(BaseType, metaclass=ABCMeta):
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
    def assign(cls, value: str, scope: Scope) -> ImmediateResult: ...


class NumericalType(PrimitiveType, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def negative(cls, a: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def add(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def subtract(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def multiply(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def divide(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def gt(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def lt(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ge(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def le(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def eq(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ne(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    unary_operators: dict = {
        **PrimitiveType.unary_operators,
        TokenKind.MINUS: "negative",
    }
    binary_operators = {
        **PrimitiveType.binary_operators,
        TokenKind.AS: "cast",
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


class Integer(NumericalType, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def logical_not(cls, a: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def remainder(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_and(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_or(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_xor(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def left_shift(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def right_shift(cls, a: str, b: str, scope: Scope) -> ImmediateResult: ...

    unary_operators: dict = {
        **NumericalType.unary_operators,
        TokenKind.BANG: "logical_not",
    }

    binary_operators = {
        **NumericalType.binary_operators,
        TokenKind.L_AND: None,
        TokenKind.L_OR: None,
        TokenKind.PERCENT: "remainder",
        TokenKind.AMPERSAND: "bit_and",
        TokenKind.V_BAR: "bit_or",
        TokenKind.CARET: "bit_xor",
        TokenKind.SHIFT_L: "left_shift",
        TokenKind.SHIFT_R: "right_shift",
    }


class Float(NumericalType, metaclass=ABCMeta):
    pass


class I32(Integer, metaclass=ABCMeta):
    final_type = True
    size = 4

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: Scope): ...


class I64(Integer, metaclass=ABCMeta):
    final_type = True
    size = 8

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: Scope): ...


class F32(Float, metaclass=ABCMeta):
    final_type = True
    size = 4

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: Scope): ...


class F64(Float, metaclass=ABCMeta):
    final_type = True
    size = 8

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: Scope): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: Scope): ...
