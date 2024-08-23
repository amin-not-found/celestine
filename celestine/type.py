from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional, Type

from scope import Scope, BaseType


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


class PrimitiveType(BaseType, metaclass=ABCMeta):
    pass


class NumericalType(PrimitiveType, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def assign(cls, value: str, scope: Scope) -> ImmediateResult: ...

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
