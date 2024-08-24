from typing import Type
from abc import ABCMeta, abstractmethod

from types_abc import TypeABC, I32ABC, I64ABC, F32ABC, F64ABC


class BaseType(metaclass=ABCMeta):
    @property
    @staticmethod
    @abstractmethod
    def size() -> int: ...

    @property
    @staticmethod
    @abstractmethod
    def abc_type() -> Type[TypeABC]: ...


class PrimitiveType(BaseType, metaclass=ABCMeta):
    pass


class NumericalType(PrimitiveType, metaclass=ABCMeta):
    pass


class IntegerType(NumericalType, metaclass=ABCMeta):
    pass


class I32(IntegerType):
    size = 4
    abc_type = I32ABC

    def __str__(self):
        return "i32"


class I64(IntegerType):
    size = 8
    abc_type = I64ABC

    def __str__(self):
        return "i64"


class F32(NumericalType):
    size = 4
    abc_type = F32ABC

    def __str__(self):
        return "f32"


class F64(NumericalType):
    size = 8
    abc_type = F64ABC

    def __str__(self):
        return "f64"


class Pointer(PrimitiveType):
    size = 8
    abc_type = I64
    contained_type: Type[BaseType]

    def __init__(self, contained_type: Type[BaseType]) -> None:
        super().__init__()
        self.contained_type = contained_type

    def __str__(self):
        return f"pointer of {self.contained_type}"
