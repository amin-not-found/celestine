from __future__ import annotations
from enum import Enum, auto
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from types_info import BaseType


class IncrementalGen:
    def __init__(self, prefix: str) -> None:
        self.counter = -1
        self.prefix = prefix

    def __call__(self) -> str:
        self.counter += 1
        return f"{self.prefix}{self.counter}"


class ScopeType(Enum):
    GLOBAL = auto()
    FUNC = auto()
    FUNC_ARG = auto()
    BLOCK = auto()


class VariableState(NamedTuple):
    mutable: bool
    address: str
    level: ScopeType
    type: BaseType


class VariableRedeclareError(Exception):
    pass


class Scope:
    _vars: dict[str, VariableState]

    def __init__(self, kind: ScopeType, parent: Scope | None = None):
        if kind != ScopeType.GLOBAL and parent is None:
            raise RuntimeError("Non global scopes should have a parent")

        self._vars = dict()
        self._kind = kind
        self._parent = parent
        self._var_gen = IncrementalGen("V")
        self._temp_gen = IncrementalGen("t")
        self._label_gen = IncrementalGen("L")

    def __repr__(self) -> str:
        if self._kind == ScopeType.GLOBAL:
            return "GLOBAL"
        return f"{self._kind}<{self._parent}>"

    @property
    def kind(self):
        return self._kind

    @property
    def parent(self):
        return self._parent

    def vars(self):
        return self._vars.items()

    def temp(self) -> str:
        if self._kind == ScopeType.GLOBAL:
            raise AttributeError(
                "Global scope can't have temporaries"
                "and couldn't find any parent function scope"
            )

        if self._kind != ScopeType.FUNC:
            return self._parent.temp()

        return self._temp_gen()

    def label(self) -> str:
        if self._kind == ScopeType.GLOBAL:
            raise AttributeError(
                "Global scope can't have block labels"
                "and couldn't find any parent function scope"
            )

        if self._kind != ScopeType.FUNC:
            return self._parent.label()

        return self._label_gen()

    def _var(self):
        match self._kind:
            case ScopeType.GLOBAL:
                raise ValueError("Non-existent or global(not supported yet) variable")
            case ScopeType.FUNC:
                return self._var_gen()
            case ScopeType.BLOCK:
                return self._parent._var()  # noqa: SLF001
            case _:
                raise ValueError("Unreachable")

    def declare_var(self, name: str, typ: BaseType, mut: bool):
        if name in self._vars:
            raise VariableRedeclareError
        self._vars[name] = VariableState(mut, self._var(), self._kind, typ)

    def declare_arg(self, name: str, typ: BaseType, mut: bool):
        self._vars[name] = VariableState(mut, self._var(), ScopeType.FUNC_ARG, typ)

    def var_state(self, name: str) -> VariableState | None:
        state = self._vars.get(name)

        if state is not None:
            return state

        if self._parent is not None:
            return self._parent.var_state(name)

        return None

    def var_signature(self, name: str):
        var = self.var_state(name)
        match var.level:
            case ScopeType.FUNC | ScopeType.BLOCK | ScopeType.FUNC_ARG:
                return f"%{var.address}"
            case ScopeType.GLOBAL:
                return NotImplementedError("Global variables not implemented")
            case _:
                raise ValueError(
                    "Not a local or global variable. \
                    This should've been caught while parsing. "
                )
