from abc import abstractmethod, ABCMeta

from lexer import TokenKind, Lexer
from errors import CompilerError, VariableRedeclareError
from scope import Scope, ScopeType
from gen import GenBackend, ImmediateResult, IfArm


class AST:
    __metaclass__ = ABCMeta  # pylint only works correctly like this

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def to_ir(self, gen: GenBackend) -> ImmediateResult:
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

    def to_ir(self, gen: GenBackend):
        return gen.int_literal(self.scope, self.lexeme)

    def __repr__(self) -> str:
        return f"IntLiteral({self.lexeme})"


class UnaryOp(AST):
    def __init__(self, op: TokenKind, expr: "Expr", scope: Scope):
        self.op = op
        self.expr = expr
        self.scope = scope

    def __repr__(self) -> str:
        return f"UnaryOp(op={self.op}, expr={self.expr})"

    def to_ir(self, gen: GenBackend) -> ImmediateResult:
        return gen.unary_op(self.scope, self.op, self.expr.to_ir(gen))


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

    def to_ir(self, gen: GenBackend):
        right = self.right.to_ir(gen)
        match self.op:
            case TokenKind.ASSIGNMENT:
                return gen.assignment(self.scope, self.left.name, right)
            case TokenKind.L_AND | TokenKind.L_OR:
                return gen.logical_connective(
                    self.scope, self.op, self.left.to_ir(gen), right
                )
            case _:
                return gen.binary_op(self.scope, self.op, self.left.to_ir(gen), right)


class IfExpr(ASTParser):
    arms: list[tuple["Expr", "Block"]]

    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = Scope(ScopeType.BLOCK, scope)

        self.arms = []

        while True:
            lexer.expect_token(TokenKind.IF)
            expr = Expr(lexer, scope)
            body = Block(lexer, self.scope)
            self.arms.append((expr, body))

            if lexer.peek().kind != TokenKind.ELSE:
                return

            lexer.next()

            if lexer.peek().kind != TokenKind.IF:
                # Final else clause
                self.arms.append((None, Block(lexer, self.scope)))
                return

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


class WhileExpr(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        self.scope = Scope(ScopeType.BLOCK, scope)

        lexer.expect_token(TokenKind.WHILE)
        self.cond = Expr(lexer, scope)
        self.body = Block(lexer, self.scope)

    def __repr__(self) -> str:
        return f"WhileExpr({self.cond}, {self.body})"

    def to_ir(self, gen: GenBackend) -> ImmediateResult:
        return gen.while_expr(self.scope, self.cond.to_ir(gen), self.body.to_ir(gen))


class Variable(AST):
    def __init__(self, name: str, scope: Scope) -> None:
        self.name = name
        self.scope = scope

    def __repr__(self) -> str:
        return f"Variable(name={self.name})"

    def to_ir(self, gen: GenBackend):
        return gen.var(self.scope, self.name)


class Expr(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        self.lexer = lexer
        self.scope = scope
        self.child = self.parse()

    def __repr__(self) -> str:
        return f"Expr({self.child})"

    def to_ir(self, gen: GenBackend):
        return self.child.to_ir(gen)

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
        self.scope = scope

    def __repr__(self) -> str:
        return f"{self.kind.name}({self.expr}),"

    def to_ir(self, gen: GenBackend):
        return gen.simple_statement(self.scope, self.kind, self.expr.to_ir(gen))


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

    def to_ir(self, gen: GenBackend):
        if self.expr is None:
            return ImmediateResult(None, "")
        return gen.var_declare(self.scope, self.name, self.expr.to_ir(gen))


class Block(ASTParser):
    def __init__(self, lexer: Lexer, scope: Scope):
        """Note: Constructor uses given scope as block scope
        and doesn't generate a new scope"""
        self.scope = scope
        self.body: Statement = []

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

    def to_ir(self, gen: GenBackend):
        return gen.block(self.scope, [stmt.to_ir(gen) for stmt in self.body])


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
        return f"Statement({self.child})"

    def to_ir(self, gen: GenBackend):
        return self.child.to_ir(gen)


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

    def to_ir(self, gen: GenBackend):
        return gen.function(self.scope, self.name, self.body.to_ir(gen))


class Program(ASTParser):
    main: Function

    def __init__(self, lexer: Lexer, scope: Scope):
        self.main = Function(lexer, scope)
        self.scope = scope

    def __repr__(self) -> str:
        return f"Program({self.main})"

    def to_ir(self, gen: GenBackend):
        return gen.program(self.scope, self.main.to_ir(gen))
