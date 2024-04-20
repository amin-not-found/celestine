from errors import CompilerError, VariableRedeclareError
from lexer import Lexer, TokenKind
from scope import Scope, ScopeType
import ast_nodes as ast


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

    def expr_precedence(self, token):
        return self.expr_precedences.get(token.kind, 0)

    def expr(self, scope: Scope, precedence=0):
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

        left = prefix(self, scope)
        token = self.lexer.peek()

        while precedence < self.expr_precedence(token):
            left = self.parse_infix(token.kind, left, scope)
            token = self.lexer.peek()

        return left

    def parse_integer(self, scope: Scope):
        token = self.lexer.next()
        return ast.IntLiteral(token.lexeme, scope)

    def parse_variable(self, scope: Scope):
        # TODO : error if uninitialized
        token = self.lexer.next()
        name = token.lexeme
        if scope.var_state(name) is None:
            raise CompilerError(
                f"Undefined identifier {name}",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )
        return ast.Variable(name, scope)

    def parse_unary_op(self, scope: Scope):
        op = self.lexer.next()
        expr = self.expr(scope, len(self.expr_precedences))
        return ast.UnaryOp(op.kind, expr, scope)

    def parse_group(self, scope: Scope):
        self.lexer.next()
        result = self.expr(scope)
        self.lexer.expect_token(TokenKind.RIGHT_PAREN)
        return result

    def parse_if(self, scope: Scope):
        if_scope = Scope(ScopeType.BLOCK, scope)
        arms = []

        while True:
            self.lexer.expect_token(TokenKind.IF)
            expr = self.expr(scope)
            body = self.block(if_scope)
            arms.append((expr, body))

            if self.lexer.peek().kind != TokenKind.ELSE:
                break

            self.lexer.next()

            if self.lexer.peek().kind != TokenKind.IF:
                # Final else clause
                arms.append((None, self.block(if_scope)))
                break
        return ast.IfExpr(arms, if_scope)

    def parse_while(self, scope: Scope):
        self.lexer.expect_token(TokenKind.WHILE)
        cond = self.expr(scope)
        scope = Scope(ScopeType.BLOCK, scope)
        body = self.block(scope)
        return ast.WhileExpr(cond, body, scope)

    def parse_infix(self, op: TokenKind, left, scope: Scope):
        self.lexer.next()
        right = self.expr(scope, self.expr_precedences[op])
        if op == TokenKind.ASSIGNMENT and not isinstance(left, ast.Variable):
            raise CompilerError(
                "Left side of assignment should be an identifier",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )
        return ast.BinaryOp(op, left, right, scope)

    def simple_statement(self, scope: Scope):
        token = self.lexer.next()
        expr = self.expr(scope)
        return ast.SimpleStatement(token.kind, expr, scope)

    def var_declare(self, scope: Scope):
        self.lexer.expect_token(TokenKind.LET)
        token = self.lexer.next()
        if token.kind != TokenKind.IDENTIFIER:
            raise CompilerError(
                "Expected identifier",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )
        name = token.lexeme

        try:
            scope.declare_var(name)
        except VariableRedeclareError as _:
            raise CompilerError(
                f"Redefining variable {name}",
                self.lexer.text,
                token.offset,
                filepath=self.lexer.path,
            )

        token = self.lexer.peek()

        expr = None
        if token.kind == TokenKind.ASSIGNMENT:
            self.lexer.next()
            expr = self.expr(scope)

        return ast.VariableDeclare(expr, name, scope)

    def parse(self):
        global_scope = Scope(ScopeType.GLOBAL)
        return self.program(global_scope)

    def block(self, scope: Scope):
        """Note: this uses given scope as block scope
        and doesn't generate a new scope"""
        body: list[ast.Statement] = []

        if scope.kind == ScopeType.BLOCK and scope.parent.kind == ScopeType.GLOBAL:
            raise CompilerError(
                "Can't have code blocks inside global scope",
                self.lexer.text,
                self.lexer.offset,
                filepath=self.lexer.path,
            )

        self.lexer.expect_token(TokenKind.LEFT_BRACE)

        while self.lexer.peek().kind != TokenKind.RIGHT_BRACE:
            body.append(self.statement(scope))

        self.lexer.expect_token(TokenKind.RIGHT_BRACE)
        return ast.Block(body, scope)

    def statement(self, scope: Scope) -> ast.Statement:
        token = self.lexer.peek()

        match token.kind:
            case TokenKind.PUTCHAR | TokenKind.RETURN:
                stmt = self.simple_statement(scope)
            case TokenKind.LET:
                stmt = self.var_declare(scope)
            case _:
                stmt = self.expr(scope)

        if not (isinstance(stmt, ast.Expr) and isinstance(stmt, self.statement_exprs)):
            self.lexer.expect_token(TokenKind.SEMICOLON)
        return stmt

    def function(self, scope: Scope):
        scope = Scope(ScopeType.FUNC, scope)

        # For now we only parse a main function that has no arguments
        self.lexer.expect_token(TokenKind.FUNCTION)
        name = self.lexer.expect_token(TokenKind.IDENTIFIER).lexeme
        assert name == "main"
        self.lexer.expect_token(TokenKind.LEFT_PAREN)
        self.lexer.expect_token(TokenKind.RIGHT_PAREN)
        self.lexer.expect_token(TokenKind.COLON)
        self.lexer.expect_token(TokenKind.INT)
        self.lexer.expect_token(TokenKind.ASSIGNMENT)

        body = self.block(scope)
        return ast.Function(name, body, scope)

    def program(self, scope: Scope):
        return ast.Program(self.function(scope), scope)

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
