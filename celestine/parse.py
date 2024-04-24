from typing import NamedTuple, TextIO, Optional

from lexer import Lexer, TokenKind, Token, UnrecognizedToken
from scope import Scope, ScopeType, VariableRedeclareError
import ast_nodes as ast
from type import I32, PrimitiveType


class Diagnostic(NamedTuple):
    msg: str
    io: TextIO
    offset: int

    def __repr__(self) -> str:
        (line, col) = self.get_pos()
        location = f"{self.io.name}:{line}:{col}"

        return f"{location}: Error: {self.msg}"

    def get_pos(self):
        offset = 0
        line = 1
        col = 1

        self.io.seek(0)
        while offset < self.offset:
            c = self.io.read(1)
            offset += 1

            if c == "\n":
                line += 1
                col = 1
            else:
                col += 1

        return (line, col)


class ParseError(Exception):
    pass


class Parser:
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
    }

    statement_exprs = (
        ast.IfExpr,
        ast.WhileExpr,
    )

    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer
        self.diagnostics: list[Diagnostic] = []
        self._last_token: Optional[Token] = None
        self._peeked = False

    def advance(self) -> Token:
        if self._peeked:
            self._peeked = False
            return self._last_token
        while True:
            try:
                self._last_token = self.lexer.next()
                return self._last_token
            except UnrecognizedToken as e:
                self.sync_error(
                    self.lexer.text, f"Syntax Error: Unexpected '{e.text}'."
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
        self.diagnostics.append(Diagnostic(msg, self.lexer.text, offset))
        self.synchronize()
        raise ParseError()

    def expect_token(self, kind: TokenKind):
        token = self.advance()
        if token.kind != kind:
            return self.sync_error(
                token.offset,
                f"Expected '{kind.name.lower()}' but got '{token.lexeme}'.",
            )
        return token

    def expr_precedence(self, token):
        return self.expr_precedences.get(token.kind, 0)

    def expr(self, scope: Scope, precedence=0):
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

    def integer(self, scope: Scope):
        token = self.advance()
        return ast.IntLiteral(token.lexeme, scope)

    def variable(self, scope: Scope):
        # TODO : error if uninitialized
        token = self.advance()
        name = token.lexeme
        state = scope.var_state(name)

        if state is None:
            return self.sync_error(token.offset, f"Undefined identifier '{name}'.")
        mut = state.mutable
        typ = state.type

        return ast.Variable(name, scope, mut, typ)

    def unary_op(self, scope: Scope):
        op = self.advance()
        expr = self.expr(scope, len(self.expr_precedences))

        if not issubclass(expr.type, PrimitiveType):
            self.error(expr.offset, "Operators are only supported for primitive types")

        if op.kind not in expr.type().unary_operators:
            self.error(
                op.offset, f"Operator {op.kind.name} is not supported by {expr.type}"
            )

        return ast.UnaryOp(op.kind, expr, scope)

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
        if not isinstance(left, ast.Variable):
            self.error(op.offset, "Left side of assignment should be an identifier.")

        var_state = scope.var_state(left.name)

        if not var_state:
            self.error(op.offset, f"Assigning to undeclared variable {left.name}.")
        elif not var_state.mutable:
            self.error(op.offset, f"Cannot assign to immutable variable {left.name}")

        right = self.expr(scope, self.expr_precedences[op.kind])
        return ast.BinaryOp(op.kind, left, right, scope)

    def infix(self, op: Token, left: ast.Expr, scope: Scope):
        self.advance()

        if op.kind == TokenKind.ASSIGNMENT:
            return self.assignment(op, left, scope)

        right = self.expr(scope, self.expr_precedences[op.kind])

        if not issubclass(left.type, PrimitiveType):
            self.error(op.offset, "Operators are only supported for primitive types.")

        if left.type != right.type:
            self.error(
                op.offset,
                f"Left({left.type}) and right({right.type})"
                " side of operation don't have the same type.",
            )

        if op.kind not in left.type.binary_operators:
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
            return self.error(token.offset, "Expected an identifier.")
        name = token.lexeme

        expr = None
        token = self.peek()
        if token.kind == TokenKind.ASSIGNMENT:
            self.advance()
            expr = self.expr(scope)
            typ = expr.type
        else:
            # TODO : WARNING : temporary workaround for types
            typ = I32

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

        while self.peek().kind != TokenKind.RIGHT_BRACE:
            body.append(self.statement(scope))

        self.expect_token(TokenKind.RIGHT_BRACE)
        return ast.Block(body, scope)

    def statement(self, scope: Scope) -> ast.Statement:
        token = self.peek()

        try:
            match token.kind:
                case TokenKind.PUTCHAR | TokenKind.RETURN:
                    stmt = self.simple_statement(scope)
                case TokenKind.LET:
                    stmt = self.var_declare(scope)
                case _:
                    stmt = self.expr(scope)
        except ParseError:
            return None

        if not (isinstance(stmt, ast.Expr) and isinstance(stmt, self.statement_exprs)):
            self.expect_token(TokenKind.SEMICOLON)
        return stmt

    def function(self, scope: Scope):
        scope = Scope(ScopeType.FUNC, scope)

        # For now we only parse a main function that has no arguments
        # TODO : handle errors when function parsing became more sophisticated
        self.expect_token(TokenKind.FUNCTION)
        name = self.expect_token(TokenKind.IDENTIFIER).lexeme
        assert name == "main"
        self.expect_token(TokenKind.LEFT_PAREN)
        self.expect_token(TokenKind.RIGHT_PAREN)
        self.expect_token(TokenKind.COLON)
        self.expect_token(TokenKind.INT)
        self.expect_token(TokenKind.ASSIGNMENT)

        body = self.block(scope)
        return ast.Function(name, body, scope, I32)

    def program(self, scope: Scope):
        try:
            return ast.Program(self.function(scope), scope)
        except ParseError:
            return None

    prefix_parse_functions = {
        TokenKind.INTEGER: integer,
        TokenKind.LEFT_PAREN: group,
        TokenKind.MINUS: unary_op,
        TokenKind.BANG: unary_op,
        TokenKind.IDENTIFIER: variable,
        TokenKind.IF: if_expr,
        TokenKind.WHILE: while_expr,
    }
