from __future__ import annotations
from typing import NamedTuple, TextIO

from lexer import Lexer, TokenKind, Token, UnrecognizedTokenError
from scope import Scope, ScopeType, VariableRedeclareError
from types_info import BaseType
import ast_nodes as ast
from types_info import (
    PrimitiveType,
    I32,
    I64,
    F32,
    F64,
    Pointer,
)


expr_precedences = {
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
    TokenKind.AS: 11,
}

primitive_types: dict[str, type[PrimitiveType]] = {
    "i32": I32,
    "i64": I64,
    "f32": F32,
    "f64": F64,
}

numerical_types = (I32, I64, F32, F64, Pointer)

integral_types = (I32, I64)

numerical_bin_ops = {
    TokenKind.PLUS,
    TokenKind.MINUS,
    TokenKind.ASTERISK,
    TokenKind.SLASH,
    TokenKind.GT,
    TokenKind.LT,
    TokenKind.GE,
    TokenKind.LE,
    TokenKind.EQ,
    TokenKind.NE,
}

numerical_unary_ops = {
    TokenKind.MINUS,
}

integral_bin_ops = {
    TokenKind.L_AND,
    TokenKind.L_OR,
    TokenKind.PERCENT,
    TokenKind.AMPERSAND,
    TokenKind.V_BAR,
    TokenKind.CARET,
    TokenKind.SHIFT_L,
    TokenKind.SHIFT_R,
}

integral_unary_ops = {
    TokenKind.BANG,
}


class Diagnostic(NamedTuple):
    msg: str
    text: TextIO
    offset: int

    def __repr__(self) -> str:
        (line, col) = self.get_pos()
        location = f"{self.text.name}:{line}:{col}"

        return f"{location}: Error: {self.msg}"

    def get_pos(self):
        offset = 0
        line = 1
        col = 1

        self.text.seek(0)
        while offset < self.offset:
            char = self.text.read(1)
            offset += 1

            if char == "\n":
                line += 1
                col = 1
            else:
                col += 1

        return (line, col)


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer
        self.diagnostics: list[Diagnostic] = []
        self.definitions: ast.Definitions = dict()
        self._last_token: Token | None = None
        self._peeked = False

    def advance(self) -> Token:
        if self._peeked:
            self._peeked = False
            return self._last_token
        while True:
            try:
                self._last_token = self.lexer.next()
                return self._last_token
            except UnrecognizedTokenError as err:  # noqa: PERF203
                self.sync_error(
                    self.lexer.offset, f"Syntax Error: Unexpected '{err.text}'."
                )

    def peek(self) -> Token:
        self._last_token = self.advance()
        self._peeked = True
        return self._last_token

    def synchronize(self):
        while True:
            token = self.advance()
            if token.kind == TokenKind.SEMICOLON:
                return
            if self.peek() in {
                TokenKind.PUTCHAR,
                TokenKind.FUNCTION,
                TokenKind.RETURN,
                TokenKind.LET,
                TokenKind.ELSE,
                TokenKind.WHILE,
            }:
                return

    def error(self, offset: int, msg: str) -> None:
        self.diagnostics.append(Diagnostic(msg, self.lexer.text, offset))

    def sync_error(self, offset: int, msg: str) -> None:
        self.error(offset, msg)
        self.synchronize()
        raise ParseError

    def expect_token(self, kind: TokenKind):
        token = self.advance()
        if token.kind != kind:
            return self.sync_error(
                token.offset,
                f"Expected '{kind.name.lower()}' but got '{token.lexeme}'.",
            )
        return token

    def expr_precedence(self, token: TokenKind):
        return expr_precedences.get(token.kind, 0)

    def expr(self, scope: Scope, precedence: int = 0):
        """An implementation of a Pratt parser"""

        token = self.peek()

        prefix = self.prefix_parse_functions.get(token.kind)
        if not prefix:
            self.sync_error(
                token.offset, f"Unexpected '{token.lexeme}' while parsing expression."
            )

        left = prefix(self, scope)
        token = self.peek()

        while precedence < self.expr_precedence(token):
            left = self.infix(token, left, scope)
            token = self.peek()

        return left

    def type(self):
        token = self.advance()

        if token.kind == TokenKind.AMPERSAND:
            return Pointer(self.type())

        if token.kind != TokenKind.IDENTIFIER:
            self.sync_error(token.offset, "Expected a type identifier.")

        typ = primitive_types.get(token.lexeme)

        if typ is None:
            self.sync_error(token.offset, f"Unknown type {token.lexeme}.")
        return typ()

    def integer(self, scope: Scope):
        token = self.advance()
        return ast.IntLiteral(token.lexeme, scope)

    def float(self, scope: Scope):
        token = self.advance()
        return ast.FloatLiteral(token.lexeme, scope)

    def identifier(self, scope: Scope):
        token = self.peek()
        name = token.lexeme

        var_state = scope.var_state(name)
        if var_state is not None:
            return self.variable(scope)

        definition = self.definitions.get(name)

        if definition is None:
            self.sync_error(token.offset, f"Undefined identifier '{name}'.")

        match definition.kind:
            case ast.DefinitionKind.FUNC:
                return self.func_call(scope)
            case _:
                raise ValueError(f"Can't parse {definition.kind}")

    def variable(self, scope: Scope):
        # TODO : error if uninitialized
        token = self.advance()
        name = token.lexeme
        state = scope.var_state(name)
        return ast.Variable(name, scope, state.type)

    def func_params(self, scope: Scope):
        params: list[ast.Expr] = []

        if self.peek().kind == TokenKind.RIGHT_PAREN:
            return params

        while True:
            params.append(self.expr(scope))
            token = self.peek()
            match token.kind:
                case TokenKind.RIGHT_PAREN:
                    break
                case TokenKind.COMMA:
                    self.advance()
                    if self.peek().kind == TokenKind.RIGHT_PAREN:
                        break
                    continue
                case _:
                    self.sync_error(
                        token.offset,
                        "Expected comma or right parenthesis"
                        " at end of function parameter.",
                    )

        return params

    def func_call(self, scope: Scope):
        name_token = self.expect_token(TokenKind.IDENTIFIER)
        self.expect_token(TokenKind.LEFT_PAREN)

        params = self.func_params(scope)
        self.expect_token(TokenKind.RIGHT_PAREN)

        definition: ast.Function = self.definitions[name_token.lexeme].body
        args = definition.arguments

        if len(params) != len(args):
            self.error(
                name_token.offset,
                "Incorrect number of arguments supplied "
                f"while calling {name_token.lexeme}",
            )

        for i, (param, arg) in enumerate(zip(params, args)):
            if type(param.type) is not type(arg[1]):
                self.error(
                    name_token.offset,
                    f"Incorrect argument type for argument number {i+1} "
                    f"while calling {name_token.lexeme}",
                )

        return ast.FuncCall(name_token.lexeme, params, scope, definition.type)

    def unary_op(self, scope: Scope):
        op = self.advance()
        expr = self.expr(scope, len(expr_precedences))

        legal_numerical_op = isinstance(expr.type, numerical_types) and (
            op.kind in numerical_unary_ops
        )
        legal_integral_op = isinstance(expr.type, integral_types) and (
            op.kind in integral_unary_ops
        )

        if not (legal_numerical_op or legal_integral_op):
            self.error(
                op.offset, f"Operator {op.kind.name} is not supported by {expr.type}"
            )

        return ast.UnaryOp(op.kind, expr, scope)

    def address(self, scope: Scope):
        token = self.advance()
        identifier = self.identifier(scope)

        if not isinstance(identifier, ast.Variable):
            raise NotImplementedError(
                "Address operator is only implemented for variables"  # TODO
            )

        var_state = scope.var_state(identifier.name)
        if not var_state.mutable:
            self.error(
                token.offset,
                "Cannot take address of immutable variable and"
                f' "{identifier.name}" isn\'t mutable.',
            )

        return ast.Address(identifier, scope)

    def dereference(self, scope: Scope):
        token = self.advance()
        expr = self.expr(scope, len(expr_precedences))
        if not isinstance(expr.type, Pointer):
            self.error(
                token.offset, f"Expected pointer for dereferencing but got {expr.type}"
            )

        return ast.Dereference(expr, scope)

    def group(self, scope: Scope):
        self.advance()
        result = self.expr(scope)
        self.expect_token(TokenKind.RIGHT_PAREN)
        return result

    def if_expr(self, scope: Scope):
        if_scope = Scope(ScopeType.BLOCK, scope)
        arms = []

        while True:
            self.expect_token(TokenKind.IF)
            expr = self.expr(scope)
            body = self.block(if_scope)
            arms.append((expr, body))

            if self.peek().kind != TokenKind.ELSE:
                break

            self.advance()

            if self.peek().kind != TokenKind.IF:
                # Final else clause
                arms.append((None, self.block(if_scope)))
                break
        return ast.IfExpr(arms, if_scope)

    def while_expr(self, scope: Scope):
        self.expect_token(TokenKind.WHILE)
        cond = self.expr(scope)
        scope = Scope(ScopeType.BLOCK, scope)
        body = self.block(scope)
        return ast.WhileExpr(cond, body, scope)

    def assignment(self, op: Token, left: ast.Expr, scope: Scope):
        if isinstance(left, ast.Variable):
            var_state = scope.var_state(left.name)
            if not var_state:
                raise ValueError(
                    "Unreachable: should have been caught while parsing variable"
                )
            if not var_state.mutable:
                self.error(
                    op.offset, f"Cannot assign to immutable variable {left.name}"
                )
        # Left should be variable or dereference
        elif not isinstance(left, ast.Dereference):
            self.sync_error(
                op.offset,
                f"Expected variable or dereference on left side of assignment."
                f" Can't assign to {left}.",
            )

        right = self.expr(scope, expr_precedences[op.kind])
        return ast.BinaryOp(op.kind, left, right, scope)

    def cast(self, op: Token, expr: ast.Expr, scope: Scope):
        typ = self.type()

        if not isinstance(expr.type, numerical_types):
            self.error(
                op.offset, "Casting is only supported for numerical primitive types."
            )

        if not isinstance(expr.type, numerical_types):
            self.error(op.offset, "Casting to non numerical type.")

        if type(typ) is type(expr.type):
            self.error(op.offset, "Pointless casting to the same type.")

        return ast.Cast(expr, scope, typ)

    def infix(self, op: Token, left: ast.Expr, scope: Scope):
        self.advance()

        match op.kind:
            case TokenKind.ASSIGNMENT:
                return self.assignment(op, left, scope)
            case TokenKind.AS:
                return self.cast(op, left, scope)

        right = self.expr(scope, expr_precedences[op.kind])

        # After excluding assignment and cast,
        # we should be left with operands of same type or pointers
        if type(left.type) is type(right.type):  # noqa: SIM114
            pass
        elif isinstance(right.type, Pointer) and isinstance(left.type, (Pointer, I64)):
            pass
        elif isinstance(left.type, Pointer) and isinstance(right.type, (Pointer, I64)):
            # Ensure right side has type of pointer for defining type of expr
            left, right = right, left

        else:
            self.error(
                op.offset,
                f"Left({left.type}) and right({right.type})"
                " side of operation don't have the same type.",
            )

        legal_numerical_op = isinstance(left.type, numerical_types) and (
            op.kind in numerical_bin_ops
        )
        legal_integral_op = isinstance(left.type, integral_types) and (
            op.kind in integral_bin_ops
        )

        if not (legal_numerical_op or legal_integral_op):
            self.error(
                op.offset, f"Operator {op.kind.name} isn't supported for {left.type}"
            )

        return ast.BinaryOp(op.kind, left, right, scope)

    def simple_statement(self, scope: Scope):
        token = self.advance()
        expr = self.expr(scope)
        return ast.SimpleStatement(token.kind, expr, scope)

    def var_declare(self, scope: Scope):
        self.expect_token(TokenKind.LET)

        mutable = False
        token = self.peek()
        if token.kind == TokenKind.MUT:
            self.advance()
            mutable = True

        token = self.advance()
        if token.kind != TokenKind.IDENTIFIER:
            return self.sync_error(token.offset, "Expected an identifier.")
        name = token.lexeme

        annotated_type = None
        token = self.peek()
        if token.kind == TokenKind.COLON:
            self.advance()
            annotated_type = self.type()

        expr = None
        expr_type = None
        token = self.peek()
        if token.kind == TokenKind.ASSIGNMENT:
            self.advance()
            expr = self.expr(scope)
            expr_type = expr.type

        if expr is None and not mutable:
            self.error(token.offset, f"Initialized immutable variable {name}")

        typ = expr_type or annotated_type

        if typ is None:
            self.sync_error(
                token.offset, "Expected type annotation for uninitialized type"
            )
        elif (annotated_type is not None and type(annotated_type) is not type(typ)) or (
            expr_type is not None and type(expr_type) is not type(typ)
        ):
            self.error(
                token.offset,
                "Assigned value doesn't have the same type as annotation",
            )

        try:
            scope.declare_var(name, typ, mutable)
        except VariableRedeclareError as _:
            self.error(token.offset, f"Declaration of existing variable '{name}'.")

        return ast.VariableDeclare(expr, name, typ, scope)

    def parse(self):
        global_scope = Scope(ScopeType.GLOBAL)
        return self.program(global_scope)

    def block(self, scope: Scope):
        """Note: this uses given scope as block scope
        and doesn't generate a new scope"""
        body: list[ast.Statement] = []

        if scope.kind == ScopeType.BLOCK and scope.parent.kind == ScopeType.GLOBAL:
            return self.error(
                self.lexer.offset, "Can't have code blocks inside global scope."
            )

        self.expect_token(TokenKind.LEFT_BRACE)

        while True:
            try:
                token = self.peek()
                if token.kind == TokenKind.RIGHT_BRACE:
                    break
                body.append(self.statement(scope))
                token = self.peek()
            except ParseError:
                pass

        self.expect_token(TokenKind.RIGHT_BRACE)
        return ast.Block(body, scope)

    def statement(self, scope: Scope) -> ast.Statement:
        token = self.peek()

        match token.kind:
            case TokenKind.PUTCHAR:
                stmt = self.simple_statement(scope)
                if not isinstance(stmt.expr.type, I32):
                    self.error(token.offset, "Putchar only accepts 32 bit integers.")
            case TokenKind.RETURN:
                stmt = self.simple_statement(scope)
            case TokenKind.LET:
                stmt = self.var_declare(scope)
            case _:
                stmt = self.expr(scope)

        self.expect_token(TokenKind.SEMICOLON)
        return stmt

    def func_args(self, scope: Scope):
        args: list[tuple[str, BaseType]] = []

        if self.peek().kind == TokenKind.RIGHT_PAREN:
            return args

        while True:
            arg = self.expect_token(TokenKind.IDENTIFIER).lexeme
            self.expect_token(TokenKind.COLON)
            typ = self.type()
            scope.declare_arg(arg, typ, True)
            args.append((arg, typ))

            token = self.peek()
            match token.kind:
                case TokenKind.RIGHT_PAREN:
                    break
                case TokenKind.COMMA:
                    self.advance()
                    if self.peek().kind == TokenKind.RIGHT_PAREN:
                        break
                    continue
                case _:
                    self.sync_error(
                        token.offset,
                        "Expected comma or right parenthesis"
                        " at end of function argument.",
                    )
        return args

    def function(self, scope: Scope):
        scope = Scope(ScopeType.FUNC, scope)

        self.expect_token(TokenKind.FUNCTION)
        name = self.expect_token(TokenKind.IDENTIFIER)
        self.expect_token(TokenKind.LEFT_PAREN)

        arguments = self.func_args(scope)
        self.expect_token(TokenKind.RIGHT_PAREN)

        self.expect_token(TokenKind.COLON)
        typ = self.type()

        # we define our function without body
        # so it's possible to call the function inside its own body
        func = ast.Function(name.lexeme, None, arguments, scope, typ)
        self.definitions[name.lexeme] = ast.Definition(func, ast.DefinitionKind.FUNC, 0)

        self.expect_token(TokenKind.ASSIGNMENT)
        func.body = self.block(scope)

        return func

    def program(self, scope: Scope):
        while True:
            try:
                self.peek()
            except StopIteration:
                break

            try:
                token = self.peek()
                func = self.function(scope)
                self.definitions[func.name] = ast.Definition(
                    func, ast.DefinitionKind.FUNC, token.offset
                )
            except ParseError:
                return None

        main = self.definitions.get("main")
        if main is None or main.kind != ast.DefinitionKind.FUNC:
            self.error(0, "Program doesn't have a main function")
        elif not isinstance(main.body.type, I32):
            self.error(main.src_offset, "Main function has to return i32.")

        return ast.Program(self.definitions, scope)

    prefix_parse_functions = {
        TokenKind.INTEGER: integer,
        TokenKind.FLOAT: float,
        TokenKind.LEFT_PAREN: group,
        TokenKind.LEFT_BRACE: block,
        TokenKind.MINUS: unary_op,
        TokenKind.BANG: unary_op,
        TokenKind.AMPERSAND: address,
        TokenKind.ASTERISK: dereference,
        TokenKind.IDENTIFIER: identifier,
        TokenKind.IF: if_expr,
        TokenKind.WHILE: while_expr,
    }
