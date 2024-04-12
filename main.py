#!/usr/bin/python3

from typing import TextIO, Optional, NamedTuple
from abc import abstractmethod, ABCMeta
from collections.abc import Iterator

from enum import Enum, auto
from dataclasses import dataclass

from sys import stderr
from os import PathLike
from pathlib import Path
from subprocess import run
from argparse import ArgumentParser, Namespace

DEBUG = True


class IncrementalGen:
    def __init__(self, prefix: str) -> None:
        self.counter = -1
        self.prefix = prefix

    def __call__(self) -> str:
        self.counter += 1
        return f"{self.prefix}{self.counter}"


class TokenKind(Enum):
    # Single-character tokens
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    SEMICOLON = auto()
    COLON = auto()
    COMMENT = auto()
    ASSIGNMENT = auto()
    PLUS = auto()
    MINUS = auto()
    ASTERISK = auto()
    SLASH = auto()
    PERCENT = auto()
    BANG = auto()
    AMPERSAND = auto()
    V_BAR = auto()
    CARET = auto()
    L_AND = auto()
    L_OR = auto()
    SHIFT_L = auto()
    SHIFT_R = auto()
    GT = auto()
    LT = auto()
    GE = auto()
    LE = auto()
    EQ = auto()
    NE = auto()
    # One or two character tokens
    # Literals
    IDENTIFIER = auto()
    INTEGER = auto()
    # Keywords
    PUTCHAR = auto()  # TODO : remove in future
    FUNCTION = auto()
    INT = auto()
    RETURN = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()


@dataclass
class Token:
    kind: TokenKind
    offset: int
    lexeme: str


class CompilerError(Exception):
    def __init__(
        self,
        msg: str,
        text: TextIO,
        offset: int,
        *args: object,
        filepath: Optional[PathLike] = None,
    ) -> None:
        (line, col) = CompilerError.get_location(text, offset)
        location = f"{filepath}:{line}:{col}" if filepath else f"code:{line}:{col}"
        super().__init__(f"{location} : error: {msg}", *args)

    @staticmethod
    def get_location(
        text: TextIO,
        offset: int,
    ):
        text.seek(0)

        content = text.read(offset)
        if not content:
            return (1, 1)

        sp = content.splitlines(keepends=True)
        return (len(sp), len(sp[-1]) + 1)  # We need column value counted from one


class EndOfTokens(CompilerError, StopIteration):
    def __init__(
        self, text: TextIO, offset: int, *args: object, filepath: PathLike | None = None
    ):
        super().__init__(
            "Unexpected end of tokens", text, offset, *args, filepath=filepath
        )


class Lexer(Iterator):
    single_char_tokens = {
        "(": TokenKind.LEFT_PAREN,
        ")": TokenKind.RIGHT_PAREN,
        "{": TokenKind.LEFT_BRACE,
        "}": TokenKind.RIGHT_BRACE,
        ";": TokenKind.SEMICOLON,
        ":": TokenKind.COLON,
        "=": TokenKind.ASSIGNMENT,
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "*": TokenKind.ASTERISK,
        "/": TokenKind.SLASH,
        "%": TokenKind.PERCENT,
        "!": TokenKind.BANG,
        "&": TokenKind.AMPERSAND,
        "|": TokenKind.V_BAR,
        "^": TokenKind.CARET,
        "<": TokenKind.LT,
        ">": TokenKind.GT,
    }

    two_char_tokens: dict[TokenKind, dict[str, TokenKind]] = {
        TokenKind.SLASH: {"/": TokenKind.COMMENT},
        TokenKind.ASSIGNMENT: {"=": TokenKind.EQ},
        TokenKind.BANG: {"=": TokenKind.NE},
        TokenKind.LT: {
            "=": TokenKind.LE,
            "<": TokenKind.SHIFT_L,
        },
        TokenKind.GT: {
            "=": TokenKind.GE,
            ">": TokenKind.SHIFT_R,
        },
        TokenKind.AMPERSAND: {"&": TokenKind.L_AND},
        TokenKind.V_BAR: {"|": TokenKind.L_OR},
    }

    keywords = {
        "putchar": TokenKind.PUTCHAR,
        "function": TokenKind.FUNCTION,
        "int": TokenKind.INT,
        "return": TokenKind.RETURN,
        "let": TokenKind.LET,
        "if": TokenKind.IF,
        "else": TokenKind.ELSE,
        "while": TokenKind.WHILE,
    }

    def __init__(self, text: TextIO, path: Optional[PathLike] = None):
        self._position = 0
        self._text = text
        self._peeked = False
        self._peeked_token = None
        self.path = path
        assert self._text.seekable()

    def __iter__(self):
        return self

    def __next__(self) -> Token:
        if self._peeked:
            self._peeked = False
            return self._peeked_token

        c = self._forward_char()

        while c.isspace():
            c = self._forward_char()

        if not c:
            self._position -= 1
            raise EndOfTokens(self.text, self._position, filepath=self.path)

        kind = self.single_char_tokens.get(c)

        # handle two or more character tokens
        two_char_candids = self.two_char_tokens.get(kind)
        if two_char_candids is not None:
            new_kind = two_char_candids.get(self._peek_char())
            if new_kind is not None:
                kind = new_kind
                self._forward_char()

        if kind == TokenKind.COMMENT:
            # A line comment
            while c and c != "\n":
                c = self._forward_char()
            return self.next()

        if kind is None:
            self._back_char()
            if c.isdigit():
                return self.parse_int()
            elif c.isalpha() or c == "_":
                return self.parse_id()
            else:
                raise CompilerError(
                    f"Unrecognized token '{c}'.", self._text, self._position, self.path
                )

        return Token(kind, self._position, c)

    def __del__(self):
        self._text.close()

    @property
    def text(self):
        return self._text

    @property
    def offset(self):
        return self._position

    def _back_char(self):
        """Move back one char"""
        self._position -= 1
        self._text.seek(self._position)

    def _forward_char(self) -> str:
        """Move one char forward"""
        self._text.seek(self._position)
        self._position += 1
        return self._text.read(1)

    def _peek_char(self) -> str:
        """Look at next char without moving"""
        token = self._forward_char()
        self._back_char()
        return token

    def next(self):
        """Get next token"""
        return next(self)

    def peek(self):
        """Look at next token without moving"""
        self._peeked_token = self.next()
        self._peeked = True
        return self._peeked_token

    def parse_int(self) -> Token:
        pos = self._position
        s = []
        while (c := self._forward_char()).isdigit():
            s.append(c)
        self._back_char()
        s = "".join(s)

        return Token(TokenKind.INTEGER, pos, s)

    def parse_id(self) -> Token:
        pos = self._position
        s = []
        c = self._forward_char()
        while c.isalnum() or c == "_":
            s.append(c)
            c = self._forward_char()
        self._back_char()

        s = "".join(s)
        kind = self.keywords.get(s) or TokenKind.IDENTIFIER

        return Token(kind, pos, s)

    @staticmethod
    def lex_file(path: PathLike):
        fp = open(path, encoding="UTF-8")
        return Lexer(fp, path)

    def expect_token(self, kind: TokenKind) -> Token:
        token = self.next()
        if token.kind != kind:
            raise CompilerError(
                f"Expected '{kind}' but got {token}",
                self._text,
                self._position,
                filepath=self.path,
            )
        return token


class ScopeType(Enum):
    GLOBAL = auto()
    FUNC = auto()
    BLOCK = auto()


class VariableRedeclareError(Exception):
    pass


class VariableState(NamedTuple):
    address: str
    level: ScopeType


class Scope:
    _vars: dict[str, VariableState]

    def __init__(self, kind: ScopeType, parent=Optional["Scope"]):
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
        assert (
            self._kind != ScopeType.GLOBAL
        ), "Global scope can't have temporaries \
            and couldn't find any parent function scope"

        if not self._kind == ScopeType.FUNC:
            return self._parent.temp()

        return self._temp_gen()

    def label(self) -> str:
        assert (
            self._kind != ScopeType.GLOBAL
        ), "Global scope can't have block labels \
            and couldn't find any parent function scope"

        if not self._kind == ScopeType.FUNC:
            return self._parent.label()

        return self._label_gen()

    def _var(self):
        assert (
            self._kind != ScopeType.GLOBAL
        ), "Non-existent or global(not supported yet) variable"

        match self._kind:
            case ScopeType.GLOBAL:
                raise ValueError("Non-existent or global(not supported yet) variable")
            case ScopeType.FUNC:
                return self._var_gen()
            case ScopeType.BLOCK:
                return self._parent._var()  # pylint: disable=protected-access
            case _:
                raise ValueError("Unreachable")

    def declare_var(self, name: str):
        if name in self._vars:
            raise VariableRedeclareError()
        self._vars[name] = VariableState(self._var(), self._kind)

    def var_state(self, name: str) -> Optional[VariableState]:
        state = self._vars.get(name)

        if state is not None:
            return state

        if self._parent is not None:
            return self._parent.var_state(name)

        return None

    def var_signature(self, name: str):
        var = self.var_state(name)
        match var.level:
            case ScopeType.FUNC | ScopeType.BLOCK:
                return f"%{var.address}"
            case ScopeType.GLOBAL:
                return NotImplementedError("Global variables not implemented")
            case _:
                raise ValueError(
                    "Not a local or global variable. \
                    This should've been caught while parsing. "
                )


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


class AST:
    __metaclass__ = ABCMeta  # pylint only works correctly like this

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def to_ir(self) -> ImmediateResult:
        """Returns a tuple of possible temporary variable name
        for result of expression and the immediate representation string"""


class ASTParser(AST):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, lexer: Lexer, scope: Scope): ...


class IntLiteral(AST):
    def __init__(self, lexeme: str, scope: Scope) -> None:
        assert lexeme.isdigit()
        self.lexeme = lexeme
        self.scope = scope

    def to_ir(self):
        var_name = self.scope.temp()
        return ImmediateResult(var_name, f"\n    %{var_name} =l copy {self.lexeme}")

    def __repr__(self) -> str:
        return f"IntLiteral({self.lexeme})"


class UnaryOp(AST):
    def __init__(self, op: TokenKind, expr: "Expr", scope: Scope):
        self.op = op
        self.expr = expr
        self.scope = scope

    def __repr__(self) -> str:
        return f"UnaryOp(op={self.op}, expr={self.expr})"

    def to_ir(self) -> ImmediateResult:
        expr = self.expr.to_ir()
        match self.op:
            case TokenKind.PLUS:
                return expr
            case TokenKind.MINUS:
                new_var = self.scope.temp()
                return ImmediateResult(
                    new_var, expr.ir + f"\n    %{new_var} =l neg %{expr.var}"
                )
            case TokenKind.BANG:
                new_var = self.scope.temp()
                return ImmediateResult(
                    new_var, expr.ir + f"\n    %{new_var} =l ceql %{expr.var}, 0"
                )
            case _:
                raise TypeError(
                    f"Unary operation not implemented for token of {self.op}"
                )


class BinaryOp(AST):
    instructions = {
        TokenKind.PLUS: "add",
        TokenKind.MINUS: "sub",
        TokenKind.ASTERISK: "mul",
        TokenKind.SLASH: "div",
        TokenKind.PERCENT: "rem",
        TokenKind.AMPERSAND: "and",
        TokenKind.CARET: "xor",
        TokenKind.V_BAR: "or",
        TokenKind.SHIFT_L: "shl",
        TokenKind.SHIFT_R: "sar",
        TokenKind.LT: "csltl",
        TokenKind.GT: "csgtl",
        TokenKind.LE: "cslel",
        TokenKind.GE: "csgel",
        TokenKind.EQ: "ceql",
        TokenKind.NE: "cnel",
    }

    def __init__(self, op: TokenKind, left: "Expr", right: "Expr", scope: Scope):
        self.op = op
        self.left = left
        self.right = right
        self.scope = scope

    def __repr__(self):
        return f"BinaryOp(op={self.op}, left={self.left}, right={self.right})"

    def assignment_ir(self):
        expr = self.right.to_ir()

        assert isinstance(
            self.left, Variable
        ), "Left side of assignment should be an identifier"

        var_signature = self.scope.var_signature(self.left.name)

        return ImmediateResult(
            expr.var, expr.ir + f"\n    storel %{expr.var}, {var_signature}"
        )

    def logical_connective_ir(self):
        left: ImmediateResult = self.left.to_ir()
        right: ImmediateResult = self.right.to_ir()
        result_var = self.scope.temp()
        resume_label = self.scope.label()
        end_label = self.scope.label()

        match self.op:
            case TokenKind.L_AND:
                jmp = f"jnz %{left.var}, @{resume_label}, @{end_label}"
            case TokenKind.L_OR:
                jmp = f"jnz %{left.var}, @{end_label}, @{resume_label}"
            case _:
                raise ValueError(f"logical op {self.op} not supported")

        ir = f"""{left.ir}
    %{result_var} =l copy %{left.var}
    {jmp}
@{resume_label}{right.ir}
    %{result_var} =l copy %{right.var}
@{end_label}"""
        return ImmediateResult(result_var, ir)

    def to_ir(self):
        match self.op:
            case TokenKind.ASSIGNMENT:
                return self.assignment_ir()
            case TokenKind.L_AND | TokenKind.L_OR:
                return self.logical_connective_ir()

        left: ImmediateResult = self.left.to_ir()
        right: ImmediateResult = self.right.to_ir()
        self_var = self.scope.temp()
        instruction = self.instructions.get(self.op)
        self_ir = f"\n    %{self_var} =l {instruction} %{left.var}, %{right.var}"
        return ImmediateResult(self_var, left.ir + right.ir + self_ir)


@dataclass
class IfArm:
    cond: "Expr"
    body: "Block"
    if_label: Optional[str]
    start_label: Optional[str]
    else_label: Optional[str]


class IfExpr(ASTParser):
    arms: list[IfArm]

    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = Scope(ScopeType.BLOCK, scope)

        self.arms = []

        while True:
            lexer.expect_token(TokenKind.IF)
            expr = Expr(lexer, scope)
            body = Block(lexer, self.scope)
            self.arms.append(IfArm(expr, body, None, None, None))

            if lexer.peek().kind != TokenKind.ELSE:
                return

            lexer.next()

            if lexer.peek().kind != TokenKind.IF:
                # Final else clause
                self.arms.append(
                    IfArm(None, Block(lexer, self.scope), None, None, None)
                )
                return

    def __repr__(self) -> str:
        res = [""]
        for arm in self.arms:
            res.append(f"({arm.cond}, {arm.body})")
        res = "\n        ".join(res)
        return f"IfExpr({res})"

    def to_ir(self):
        assert len(self.arms) > 0

        end_label = self.scope.label()

        else_label = self.scope.label()
        for arm in self.arms:
            arm.if_label = else_label
            arm.start_label = self.scope.label()
            arm.else_label = self.scope.label()
            else_label = arm.else_label

        ir = []

        # arms: list[tuple[Optional[Expr], Block, str,str,str]] = []
        for arm in self.arms:
            ir.append(f"\n@{arm.if_label}")

            if arm.cond is None:
                # Else clause is the only arm without cond
                ir.append(arm.body.to_ir().ir)
                break

            cond = arm.cond.to_ir()
            ir.append(cond.ir)
            ir.append(f"\n    jnz %{cond.var}, @{arm.start_label}, @{arm.else_label}")
            ir.append(f"\n@{arm.start_label}")
            ir.append(arm.body.to_ir().ir)
            ir.append(f"\n    jmp @{end_label}")

        ir.append(f"\n@{end_label}")
        res_var = self.scope.temp()
        ir.append(f"\n    %{res_var} =l copy 0")
        res_ir = "".join(ir)

        return ImmediateResult(res_var, res_ir)


class WhileExpr(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = Scope(ScopeType.BLOCK, scope)

        lexer.expect_token(TokenKind.WHILE)
        self.cond = Expr(lexer, scope)
        self.body = Block(lexer, self.scope)

    def __repr__(self) -> str:
        return f"WhileExpr({self.cond}, {self.body})"

    def to_ir(self) -> ImmediateResult:
        cond = self.cond.to_ir()
        while_label = self.scope.label()
        start_label = self.scope.label()
        end_label = self.scope.label()
        res_var = self.scope.temp()
        ir = f"""
@{while_label}{cond.ir}
    jnz %{cond.var}, @{start_label}, @{end_label}
@{start_label}{self.body.to_ir().ir}
    jmp @{while_label}
@{end_label}
    %{res_var} =l copy 0"""
        return ImmediateResult(res_var, ir)


class Variable(AST):
    def __init__(self, name: str, scope: Scope) -> None:
        self.name = name
        self.scope = scope

    def __repr__(self) -> str:
        return f"Variable(name={self.name})"

    def to_ir(self):
        temp_var = self.scope.temp()
        signature = self.scope.var_signature(self.name)
        return ImmediateResult(temp_var, f"\n    %{temp_var} =l loadl {signature}")


class Expr(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        self.lexer = lexer
        self.scope = scope
        self.child = self.parse()

    def __repr__(self) -> str:
        return str(self.child)

    def to_ir(self):
        return self.child.to_ir()

    def parse(self, precedence=0):
        """An implementation of a Pratt parser"""

        token = self.lexer.peek()

        prefix = self.prefix_parse_functions.get(token.kind)
        if not prefix:
            raise CompilerError(
                f"Unexpected token '{token}' while parsing expression.",
                self.lexer.text,
                token.offset,
                filepath=self.lexer.path,
            )

        left = prefix(self)
        token = self.lexer.peek()

        while precedence < self.precedence(token):
            left = self.parse_infix(token.kind, left)
            token = self.lexer.peek()

        return left

    def precedence(self, token):
        return self.precedences.get(token.kind, 0)

    def parse_integer(self):
        token = self.lexer.next()
        return IntLiteral(token.lexeme, self.scope)

    def parse_variable(self):
        # TODO : error if uninitialized
        token = self.lexer.next()
        name = token.lexeme
        if self.scope.var_state(name) is None:
            raise CompilerError(
                f"Undefined identifier {name}",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )
        return Variable(name, self.scope)

    def parse_unary_op(self):
        op = self.lexer.next()
        # More precedence than other operators
        expr = self.parse(len(self.precedences))
        return UnaryOp(op.kind, expr, self.scope)

    def parse_group(self):
        self.lexer.next()
        result = self.parse()
        self.lexer.expect_token(TokenKind.RIGHT_PAREN)
        return result

    def parse_if(self):
        return IfExpr(self.lexer, self.scope)

    def parse_while(self):
        return WhileExpr(self.lexer, self.scope)

    def parse_infix(self, op: TokenKind, left):
        self.lexer.next()
        right = self.parse(self.precedences[op])
        if op == TokenKind.ASSIGNMENT and not isinstance(left, Variable):
            raise CompilerError(
                "Left side of assignment should be an identifier",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )
        return BinaryOp(op, left, right, self.scope)

    precedences = {
        TokenKind.ASSIGNMENT: 1,
        TokenKind.L_OR: 2,
        TokenKind.L_AND: 3,
        TokenKind.LT: 4,
        TokenKind.GT: 4,
        TokenKind.LE: 4,
        TokenKind.GE: 4,
        TokenKind.EQ: 4,
        TokenKind.NE: 4,
        TokenKind.V_BAR: 5,
        TokenKind.CARET: 6,
        TokenKind.AMPERSAND: 7,
        TokenKind.SHIFT_L: 8,
        TokenKind.SHIFT_R: 8,
        TokenKind.PLUS: 9,
        TokenKind.MINUS: 9,
        TokenKind.ASTERISK: 10,
        TokenKind.SLASH: 10,
        TokenKind.PERCENT: 10,
    }

    prefix_parse_functions = {
        TokenKind.INTEGER: parse_integer,
        TokenKind.LEFT_PAREN: parse_group,
        TokenKind.PLUS: parse_unary_op,
        TokenKind.MINUS: parse_unary_op,
        TokenKind.BANG: parse_unary_op,
        TokenKind.IDENTIFIER: parse_variable,
        TokenKind.IF: parse_if,
        TokenKind.WHILE: parse_while,
    }


class SimpleStatement(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        token = lexer.next()
        self.kind = token.kind
        self.expr = Expr(lexer, scope)

    def __repr__(self) -> str:
        return f"{self.kind.name}({self.expr}),"

    def to_ir(self):
        (var_name, res) = self.expr.to_ir()
        match self.kind:
            case TokenKind.PUTCHAR:
                res += f"\n    call $putchar(l %{var_name})"
            case TokenKind.RETURN:
                res += f"\n    ret %{var_name}"
            case _:
                raise TypeError(f"Can't use token {self.kind} as statement type")
        return ImmediateResult(None, res)


class VariableDeclare(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = scope

        lexer.expect_token(TokenKind.LET)
        token = lexer.next()
        if token.kind != TokenKind.IDENTIFIER:
            raise CompilerError(
                "Expected identifier", lexer.text, lexer.offset, filepath=lexer.path
            )
        self.name = token.lexeme

        try:
            scope.declare_var(self.name)
        except VariableRedeclareError as _:
            raise CompilerError(
                f"Redefining variable {self.name}",
                lexer.text,
                token.offset,
                filepath=lexer.path,
            )

        token = lexer.peek()
        if token.kind != TokenKind.ASSIGNMENT:
            self.expr = None
            return

        lexer.next()
        self.expr = Expr(lexer, scope)

    def __repr__(self) -> str:
        return f"VariableDeclare(name={self.name}, expr={self.expr}),"

    def to_ir(self):
        if self.expr is None:
            return ImmediateResult(None, "")
        expr = self.expr.to_ir()
        signature = self.scope.var_signature(self.name)
        ir = expr.ir + f"\n    storel %{expr.var}, {signature}"
        return ImmediateResult(None, ir)


class Block(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        """Note: Constructor uses given scope as block scope
        and doesn't generate a new scope"""
        self.scope = scope
        self.body = []

        if (
            self.scope.kind == ScopeType.BLOCK
            and self.scope.parent.kind == ScopeType.GLOBAL
        ):
            raise CompilerError(
                "Can't have code blocks inside global scope",
                lexer.text,
                lexer.offset,
                filepath=lexer.path,
            )

        lexer.expect_token(TokenKind.LEFT_BRACE)
        while lexer.peek().kind != TokenKind.RIGHT_BRACE:
            self.body.append(Statement(lexer, self.scope))
        lexer.expect_token(TokenKind.RIGHT_BRACE)

    def __repr__(self) -> str:
        return "\n    " + "\n    ".join(map(str, self.body))

    def to_ir(self):
        body = []
        body = "".join(
            f"\n    %{var.address} =l alloc8 8" for _, var in self.scope.vars()
        )
        body += "".join(stmt.to_ir().ir for stmt in self.body)
        return ImmediateResult(None, body)


class Statement(ASTParser):
    child: Expr | SimpleStatement

    statement_types = {
        TokenKind.PUTCHAR: SimpleStatement,
        TokenKind.RETURN: SimpleStatement,
        TokenKind.LET: VariableDeclare,
    }

    statement_exprs = (
        IfExpr,
        WhileExpr,
    )

    def __init__(self, lexer: Lexer, scope: Scope):
        token = lexer.peek()
        stmt_type = self.statement_types.get(token.kind)

        if not stmt_type:
            self.child = Expr(lexer, scope)
        else:
            self.child = stmt_type(lexer, scope)

        if not (
            isinstance(self.child, Expr)
            and isinstance(self.child.child, self.statement_exprs)
        ):
            lexer.expect_token(TokenKind.SEMICOLON)

    def __repr__(self) -> str:
        return str(self.child)

    def to_ir(self):
        return self.child.to_ir()


class Function(ASTParser):
    name: str

    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = Scope(ScopeType.FUNC, scope)

        # For now we only parse a main function that has no arguments
        lexer.expect_token(TokenKind.FUNCTION)
        self.name = lexer.expect_token(TokenKind.IDENTIFIER).lexeme
        assert self.name == "main"
        lexer.expect_token(TokenKind.LEFT_PAREN)
        lexer.expect_token(TokenKind.RIGHT_PAREN)
        lexer.expect_token(TokenKind.COLON)
        lexer.expect_token(TokenKind.INT)
        lexer.expect_token(TokenKind.ASSIGNMENT)

        self.body = Block(lexer, self.scope)

    def __repr__(self) -> str:
        return f"Function(name={self.name}, body={self.body}\n  )\n"

    def to_ir(self):
        top = f"\nexport function l ${self.name}() {{\n@start"
        bottom = "\n}\n"
        return ImmediateResult(None, top + self.body.to_ir().ir + bottom)


class Program(ASTParser):
    main: Function

    def __init__(self, lexer: Lexer, scope: Scope):
        self.main = Function(lexer, scope)

    def __repr__(self) -> str:
        return f"Program({self.main})"

    def to_ir(self):
        return ImmediateResult(
            None,
            """
function w $pushchar(l %c) { \n\
@start 
    %a =l alloc4 8 \n\
    storew %c, %a \n\
    %b =w call $write(w 1, l %a, w 1) \n\
    ret %b \n\
} \n"""
            + self.main.to_ir().ir,
        )


def compile_file(args: Namespace) -> Path:
    path = Path(args.file)
    ssa_file = path.parent.joinpath(f"{path.stem}.ssa")
    asm_file = path.parent.joinpath(f"{path.stem}.s")
    out_file = path.parent.joinpath(f"{path.stem}.out")

    lexer = Lexer.lex_file(path)
    global_scope = Scope(ScopeType.GLOBAL)
    program = Program(lexer, global_scope)

    if args.print:
        print(program, file=stderr)

    with open(ssa_file, "w", encoding="UTF-8") as f:
        f.write(program.to_ir().ir)

    run(
        ["qbe", "-o", asm_file, ssa_file],
        check=True,
    )

    run(["cc", asm_file, "-o", out_file], check=True)

    if not args.keep:
        ssa_file.unlink()
        asm_file.unlink()

    return out_file


def main():
    arg_parser = ArgumentParser(description="Celestine compiler.")
    arg_parser.add_argument(
        "-r", "--run", help="run executable output", action="store_true"
    )
    arg_parser.add_argument(
        "-p", "--print", help="print generated AST", action="store_true"
    )
    arg_parser.add_argument(
        "-k",
        "--keep",
        help="don't delete files from each compiling stage",
        action="store_true",
    )
    arg_parser.add_argument("file")

    args = arg_parser.parse_args()
    try:
        out_path = compile_file(args)

    except CompilerError as ce:
        if DEBUG:
            raise ce
        print(ce)
        exit(-1)

    if args.run:
        run([out_path.absolute()], check=True)


if __name__ == "__main__":
    main()
