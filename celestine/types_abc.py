from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional, TYPE_CHECKING

# prevent unnecessary circular import on execution
if TYPE_CHECKING:
    from scope import Scope


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


class TypeABC(metaclass=ABCMeta):
    pass


class PrimitiveTypeABC(TypeABC, metaclass=ABCMeta):
    pass


class NumericalABC(PrimitiveTypeABC, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def assign(cls, value: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def negative(cls, a: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def add(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def subtract(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def multiply(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def divide(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def gt(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def lt(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ge(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def le(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def eq(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def ne(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...


class IntegerABC(NumericalABC, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def logical_not(cls, a: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def remainder(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_and(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_or(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def bit_xor(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def left_shift(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...

    @classmethod
    @abstractmethod
    def right_shift(cls, a: str, b: str, scope: "Scope") -> ImmediateResult: ...


class FloatABC(NumericalABC, metaclass=ABCMeta):
    pass


class I32ABC(IntegerABC, metaclass=ABCMeta):
    final_type = True

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: "Scope"): ...


class I64ABC(IntegerABC, metaclass=ABCMeta):
    final_type = True

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: "Scope"): ...


class F32ABC(FloatABC, metaclass=ABCMeta):
    final_type = True

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f64(cls, var: str, scope: "Scope"): ...


class F64ABC(FloatABC, metaclass=ABCMeta):
    final_type = True

    @classmethod
    @abstractmethod
    def from_i32(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_i64(cls, var: str, scope: "Scope"): ...

    @classmethod
    @abstractmethod
    def from_f32(cls, var: str, scope: "Scope"): ...
