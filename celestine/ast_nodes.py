from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import NamedTuple, Type
from enum import Enum, auto

from lexer import TokenKind
from scope import Scope, BaseType
from gen import GenBackend, GenResult, IfArm, BlockIR
from types_info import PrimitiveType, NumericalType, I32, F32, Pointer


class AST(metaclass=ABCMeta):
    type: Type[BaseType] | None = None

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def to_ir(self, gen: GenBackend) -> GenResult:
        """Returns a tuple of possible temporary variable name
        for result of expression and the immediate representation string"""


class DefinitionKind(Enum):
    FUNC = auto()


class Definition(NamedTuple):
    body: AST
    kind: DefinitionKind
    src_offset: int

    def __repr__(self):
        return str(self.body)


Definitions = dict[str, Definition]


@dataclass
class Literal(AST, metaclass=ABCMeta):
    lexeme: str
    scope: Scope

    @property
    @abstractmethod
    def type(self) -> Type[PrimitiveType]: ...

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.literal(self.scope, self.type, self.lexeme)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.lexeme})"


class IntLiteral(Literal):
    type = I32()


class FloatLiteral(Literal):
    type = F32()


@dataclass
class Variable(AST):
    name: str
    scope: Scope
    type: Type[BaseType]

    def __repr__(self) -> str:
        return f"Variable(name={self.name})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.var(self.scope, self.name, self.type)


@dataclass
class Address(AST):
    expr: "Variable"
    scope: Scope

    def __post_init__(self):
        self.type: Type[BaseType] = Pointer(self.expr.type)

    def __repr__(self) -> str:
        return f"Address(expr={self.expr})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.address(self.scope, self.expr.name)


@dataclass
class Dereference(AST):
    expr: "Expr"
    scope: Scope

    def __post_init__(self):
        assert isinstance(self.expr.type, Pointer)
        self.type: Type[BaseType] = self.expr.type.contained_type

    def __repr__(self) -> str:
        return f"Dereference(expr={self.expr})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.dereference(self.scope, self.type, self.expr.to_ir(gen))


@dataclass
class Cast(AST):
    expr: "Expr"
    scope: Scope
    type: Type[NumericalType]

    def __repr__(self) -> str:
        return f"Cast(expr={self.expr}, to={self.type})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        expr = self.expr.to_ir(gen)
        return gen.cast(self.scope, expr, self.expr.type, self.type)


@dataclass
class UnaryOp(AST):
    op: TokenKind
    expr: "Expr"
    scope: Scope

    def __post_init__(self):
        self.type: Type[PrimitiveType] = self.expr.type

    def __repr__(self) -> str:
        return f"UnaryOp(op={self.op}, expr={self.expr})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        expr = self.expr.to_ir(gen)
        return gen.unary_op(self.scope, self.op, self.type, expr)


@dataclass
class BinaryOp(AST):
    op: TokenKind
    left: Variable | Dereference
    right: "Expr"
    scope: Scope

    def __post_init__(self):
        if self.op == TokenKind.ASSIGNMENT:
            assert isinstance(self.left, (Variable, Dereference))
        self.type: Type[PrimitiveType] = self.right.type

    def __repr__(self):
        return f"BinaryOp(op={self.op}, left={self.left}, right={self.right})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        right = self.right.to_ir(gen)
        match self.op:
            case TokenKind.ASSIGNMENT if isinstance(self.left, Variable):
                return gen.var_assignment(self.scope, self.left.name, self.type, right)
            case TokenKind.ASSIGNMENT if isinstance(self.left, Dereference):
                ref = self.left.expr.to_ir(gen)
                return gen.deref_assignment(self.scope, ref, self.type, right)
            case TokenKind.L_AND | TokenKind.L_OR:
                return gen.logical_connective(
                    self.scope, self.op, self.left.to_ir(gen), right, self.type
                )
            case _:
                return gen.binary_op(
                    self.scope, self.op, self.type, self.left.to_ir(gen), right
                )


@dataclass
class IfExpr(AST):
    arms: list[tuple["Expr", "Block"]]
    scope: Scope

    def __post_init__(self):
        self.type = I32()

    def __repr__(self) -> str:
        res = [""]
        for arm in self.arms:
            res.append(f"({arm[0]}, {arm[1]})")
        res = "\n        ".join(res)
        return f"IfExpr({res})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        arms = []
        for arm in self.arms:
            cond = arm[0]
            if cond is not None:
                cond = cond.to_ir(gen)
            arms.append(IfArm(cond, arm[1].to_block_ir(gen, False)))

        return gen.if_expr(self.scope, arms)


@dataclass
class WhileExpr(AST):
    cond: "Expr"
    body: "Block"
    scope: Scope

    def __post_init__(self):
        self.type = self.body.type

    def __repr__(self) -> str:
        body = str(self.body).replace("\n", "\n    ")
        return f"WhileExpr({self.cond}, {body})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.while_expr(
            self.scope,
            self.cond.to_ir(gen),
            self.body.to_block_ir(gen, False),
        )


@dataclass
class FuncCall(AST):
    function_name: str
    parameters: list["Expr"]
    scope: Scope
    type: Type[BaseType]

    def __repr__(self) -> str:
        args = ", ".join(map(str, self.parameters))
        return f"FunctionCall(function={self.function_name}, args={args})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        params = [(param.to_ir(gen), param.type) for param in self.parameters]
        return gen.func_call(self.scope, self.function_name, params, self.type)


Expr = IntLiteral | Variable | Cast | UnaryOp | BinaryOp | IfExpr | WhileExpr | FuncCall


@dataclass
class SimpleStatement(AST):
    kind: TokenKind
    expr: Expr
    scope: Scope

    def __repr__(self) -> str:
        return f"{self.kind.name}({self.expr}),"

    def to_ir(self, gen: GenBackend) -> GenResult:
        expr = self.expr.to_ir(gen)
        match self.kind:
            case TokenKind.PUTCHAR:
                return gen.putchar(expr, self.expr.type)
            case TokenKind.RETURN:
                return gen.return_stmt(expr)
            case _:
                raise ValueError("Should be unreachable")


@dataclass
class VariableDeclare(AST):
    expr: Expr
    name: str
    var_type: Type[BaseType]
    scope: Scope

    def __repr__(self) -> str:
        return f"VariableDeclare(name={self.name}, expr={self.expr}),"

    def to_ir(self, gen: GenBackend) -> GenResult:
        if self.expr is None:
            return GenResult(gen)
        return gen.var_declare(
            self.scope, self.name, self.var_type, self.expr.to_ir(gen)
        )


Statement = SimpleStatement | VariableDeclare | Expr


@dataclass
class Block(AST):
    body: list[Statement]
    scope: Scope

    def __post_init__(self):
        self.type = I32()
        self.start = self.scope.label()
        self.end = self.scope.label()

    def __repr__(self) -> str:
        body = "\n  ".join(str(stmt) for stmt in self.body)
        return "Block(\n    " + body.replace("\n  ", "\n    ") + ")"

    def to_ir(self, gen: GenBackend, gen_labels=True) -> GenResult:
        return gen.block(
            self.scope,
            self.start,
            self.end,
            [stmt.to_ir(gen) for stmt in self.body],
            gen_labels,
        )

    def to_block_ir(self, gen: GenBackend, gen_labels=True) -> BlockIR:
        return BlockIR(self.to_ir(gen, gen_labels), self.start, self.end)


@dataclass
class Function(AST):
    name: str
    body: Block
    arguments: list[tuple[str, BaseType]]
    scope: Scope
    type: Type[BaseType]

    def __repr__(self) -> str:
        args = ", ".join(f"{arg}={typ.__name__}" for arg, typ in self.arguments)
        return f"Function(name={self.name}, args=({args}), body={self.body}\n  )"

    def to_ir(self, gen: GenBackend) -> GenResult:
        return gen.function(
            self.scope,
            self.name,
            self.arguments,
            self.body.to_block_ir(gen, False),
            self.type,
        )


@dataclass
class Program(AST):
    definitions: Definitions
    scope: Scope

    def __repr__(self) -> str:
        functions = "\n  ".join(str(body) for _, body in self.definitions.items())
        return f"Program({functions})"

    def to_ir(self, gen: GenBackend) -> GenResult:
        functions = [f.body.to_ir(gen) for f in self.definitions.values()]
        return gen.program(self.scope, functions)
