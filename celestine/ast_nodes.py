from abc import abstractmethod, ABCMeta
from dataclasses import dataclass

from lexer import TokenKind
from scope import Scope, Type
from gen import GenBackend, ImmediateResult, IfArm
from type import PrimitiveType, I32


class AST:
    __metaclass__ = ABCMeta  # pylint only works correctly like this
    type: Type | None = None

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def to_ir(self, gen: GenBackend) -> ImmediateResult:
        """Returns a tuple of possible temporary variable name
        for result of expression and the immediate representation string"""


@dataclass
class IntLiteral(AST):
    lexeme: str
    scope: Scope
    type = I32

    def __post_init__(self):
        assert self.lexeme.isdigit()

    def to_ir(self, gen: GenBackend):
        return gen.int_literal(self.scope, self.lexeme)

    def __repr__(self) -> str:
        return f"IntLiteral({self.lexeme})"


@dataclass
class Variable(AST):
    name: str
    scope: Scope
    mutable: bool
    type: Type

    def __repr__(self) -> str:
        return f"Variable(name={self.name})"

    def to_ir(self, gen: GenBackend):
        return gen.var(self.scope, self.name)


@dataclass
class UnaryOp(AST):
    op: TokenKind
    expr: "Expr"
    scope: Scope

    def __post_init__(self):
        self.type: PrimitiveType = self.expr.type

    def __repr__(self) -> str:
        return f"UnaryOp(op={self.op}, expr={self.expr})"

    def to_ir(self, gen: GenBackend) -> ImmediateResult:
        expr = self.expr.to_ir(gen)
        return gen.unary_op(self.scope, self.op, self.type, expr)


@dataclass
class BinaryOp(AST):
    op: TokenKind
    left: "Expr"
    right: "Expr"
    scope: Scope

    def __post_init__(self):
        self.type: PrimitiveType = self.right.type

    def __repr__(self):
        return f"BinaryOp(op={self.op}, left={self.left}, right={self.right})"

    def to_ir(self, gen: GenBackend):
        right = self.right.to_ir(gen)
        match self.op:
            case TokenKind.ASSIGNMENT:
                return gen.assignment(self.scope, self.left.name, self.type, right)
            case TokenKind.L_AND | TokenKind.L_OR:
                return gen.logical_connective(
                    self.scope, self.op, self.left.to_ir(gen), right
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
        self.type = I32

    def __repr__(self) -> str:
        res = [""]
        for arm in self.arms:
            res.append(f"({arm.cond}, {arm.body})")
        res = "\n        ".join(res)
        return f"IfExpr({res})"

    def to_ir(self, gen: GenBackend):
        arms = []
        for arm in self.arms:
            cond = arm[0]
            if cond is not None:
                cond = cond.to_ir(gen)
            arms.append(IfArm(cond, arm[1].to_ir(gen)))

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

    def to_ir(self, gen: GenBackend) -> ImmediateResult:
        return gen.while_expr(self.scope, self.cond.to_ir(gen), self.body.to_ir(gen))


Expr = IntLiteral | Variable | UnaryOp | BinaryOp | IfExpr | WhileExpr


@dataclass
class SimpleStatement(AST):
    kind: TokenKind
    expr: Expr
    scope: Scope

    def __repr__(self) -> str:
        return f"{self.kind.name}({self.expr}),"

    def to_ir(self, gen: GenBackend):
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
    var_type: Type
    scope: Scope

    def __repr__(self) -> str:
        return f"VariableDeclare(name={self.name}, expr={self.expr}),"

    def to_ir(self, gen: GenBackend):
        if self.expr is None:
            return ImmediateResult(None, "")
        return gen.var_declare(
            self.scope, self.name, self.var_type, self.expr.to_ir(gen)
        )


Statement = SimpleStatement | VariableDeclare | Expr


@dataclass
class Block(AST):
    body: list[Statement]
    scope: Scope

    def __post_init__(self):
        self.type = I32

    def __repr__(self) -> str:
        return "\n    " + "\n    ".join(map(str, self.body))

    def to_ir(self, gen: GenBackend):
        return gen.block(self.scope, [stmt.to_ir(gen) for stmt in self.body])


@dataclass
class Function(AST):
    name: str
    body: Block
    scope: Scope
    type: Type

    def __repr__(self) -> str:
        return f"Function(name={self.name}, body={self.body}\n  )\n"

    def to_ir(self, gen: GenBackend):
        return gen.function(self.scope, self.name, self.body.to_ir(gen))


@dataclass
class Program(AST):
    main: Function
    scope: Scope

    def __repr__(self) -> str:
        return f"Program({self.main})"

    def to_ir(self, gen: GenBackend):
        return gen.program(self.scope, self.main.to_ir(gen))
