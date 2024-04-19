from typing import Optional, NamedTuple
from abc import abstractmethod, ABCMeta

from lexer import TokenKind
from scope import Scope


class ImmediateResult(NamedTuple):
    var: Optional[str]
    ir: str


class IfArm(NamedTuple):
    cond: ImmediateResult
    body: ImmediateResult


def gen_method(func):
    return staticmethod(abstractmethod(func))


class GenBackend:
    # pylint: disable=unused-argument
    __metaclass__ = ABCMeta

    @gen_method
    def int_literal(scope: Scope, integer_str: str) -> ImmediateResult: ...

    @gen_method
    def unary_op(
        scope: Scope, op: TokenKind, expr: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def assignment(
        scope: Scope, name: str, expr: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def logical_connective(
        scope: Scope, op: TokenKind, left: ImmediateResult, right: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def binary_op(
        scope: Scope, op: TokenKind, left: ImmediateResult, right: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def if_expr(scope: Scope, arms: list[IfArm]) -> ImmediateResult: ...

    @gen_method
    def while_expr(
        scope: Scope, cond: ImmediateResult, body: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def var(scope: Scope, name: str) -> ImmediateResult: ...

    @gen_method
    def simple_statement(
        scope: Scope, kind: TokenKind, expr: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def var_declare(
        scope: Scope, name: str, expr: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def block(scope: Scope, statements: list[ImmediateResult]) -> ImmediateResult: ...

    @gen_method
    def function(scope: Scope, name: str, body: ImmediateResult) -> ImmediateResult: ...

    @gen_method
    def program(scope: Scope, main_func: ImmediateResult) -> ImmediateResult: ...


class IfLabels(NamedTuple):
    if_: Optional[str]
    start: Optional[str]
    else_: Optional[str]


class QBE(GenBackend):
    bin_instructions = {
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

    @staticmethod
    def int_literal(scope: Scope, integer_str: str):
        var_name = scope.temp()
        return ImmediateResult(var_name, f"\n    %{var_name} =l copy {integer_str}")

    @staticmethod
    def unary_op(scope: Scope, op: TokenKind, expr: ImmediateResult):
        match op:
            case TokenKind.PLUS:
                return expr
            case TokenKind.MINUS:
                new_var = scope.temp()
                return ImmediateResult(
                    new_var, expr.ir + f"\n    %{new_var} =l neg %{expr.var}"
                )
            case TokenKind.BANG:
                new_var = scope.temp()
                return ImmediateResult(
                    new_var, expr.ir + f"\n    %{new_var} =l ceql %{expr.var}, 0"
                )
            case _:
                raise TypeError(f"Unary operation not implemented for token of {op}")

    @staticmethod
    def assignment(scope: Scope, name: str, expr: ImmediateResult) -> ImmediateResult:
        var_signature = scope.var_signature(name)

        return ImmediateResult(
            expr.var, expr.ir + f"\n    storel %{expr.var}, {var_signature}"
        )

    @staticmethod
    def logical_connective(
        scope: Scope, op: TokenKind, left: ImmediateResult, right: ImmediateResult
    ):
        result_var = scope.temp()
        resume_label = scope.label()
        end_label = scope.label()

        match op:
            case TokenKind.L_AND:
                jmp = f"jnz %{left.var}, @{resume_label}, @{end_label}"
            case TokenKind.L_OR:
                jmp = f"jnz %{left.var}, @{end_label}, @{resume_label}"
            case _:
                raise ValueError(f"logical op {op} not supported")

        ir = f"""{left.ir}
    %{result_var} =l copy %{left.var}
    {jmp}
@{resume_label}{right.ir}
    %{result_var} =l copy %{right.var}
@{end_label}"""
        return ImmediateResult(result_var, ir)

    @staticmethod
    def binary_op(
        scope: Scope, op: TokenKind, left: ImmediateResult, right: ImmediateResult
    ) -> ImmediateResult:
        self_var = scope.temp()
        instruction = QBE.bin_instructions.get(op)
        self_ir = f"\n    %{self_var} =l {instruction} %{left.var}, %{right.var}"
        return ImmediateResult(self_var, left.ir + right.ir + self_ir)

    @staticmethod
    def if_expr(scope: Scope, arms: list[IfArm]) -> ImmediateResult:
        assert len(arms) > 0

        arm_labels: list[IfLabels] = []
        end_label = scope.label()

        else_label = scope.label()
        for arm in arms:
            arm_labels.append(IfLabels(else_label, scope.label(), scope.label()))
            else_label = arm_labels[-1].else_

        ir = []

        for arm, labels in zip(arms, arm_labels):
            ir.append(f"\n@{labels.if_}")

            if arm.cond is None:
                # Else clause is the only arm without cond
                ir.append(arm.body.ir)
                break

            ir.append(arm.cond.ir)
            ir.append(f"\n    jnz %{arm.cond.var}, @{labels.start}, @{labels.else_}")
            ir.append(f"\n@{labels.start}")
            ir.append(arm.body.ir)
            ir.append(f"\n    jmp @{end_label}")

        ir.append(f"\n@{end_label}")
        res_var = scope.temp()
        ir.append(f"\n    %{res_var} =l copy 0")
        res_ir = "".join(ir)

        return ImmediateResult(res_var, res_ir)

    @staticmethod
    def while_expr(
        scope: Scope, cond: ImmediateResult, body: ImmediateResult
    ) -> ImmediateResult:
        while_label = scope.label()
        start_label = scope.label()
        end_label = scope.label()
        res_var = scope.temp()

        ir = f"""
@{while_label}{cond.ir}
    jnz %{cond.var}, @{start_label}, @{end_label}
@{start_label}{body.ir}
    jmp @{while_label}
@{end_label}
    %{res_var} =l copy 0"""

        return ImmediateResult(res_var, ir)

    @staticmethod
    def var(scope: Scope, name: str) -> ImmediateResult:
        temp_var = scope.temp()
        signature = scope.var_signature(name)
        return ImmediateResult(temp_var, f"\n    %{temp_var} =l loadl {signature}")

    @staticmethod
    def simple_statement(
        scope: Scope, kind: TokenKind, expr: ImmediateResult
    ) -> ImmediateResult:
        ir = expr.ir
        match kind:
            case TokenKind.PUTCHAR:
                ir += f"\n    call $putchar(l %{expr.var})"
            case TokenKind.RETURN:
                ir += f"\n    ret %{expr.var}"
            case _:
                raise TypeError(f"Can't use token {kind} as statement type")
        return ImmediateResult(None, ir)

    @staticmethod
    def var_declare(scope: Scope, name: str, expr: ImmediateResult) -> ImmediateResult:
        signature = scope.var_signature(name)
        ir = expr.ir + f"\n    storel %{expr.var}, {signature}"
        return ImmediateResult(None, ir)

    @staticmethod
    def block(scope: Scope, statements: list[ImmediateResult]) -> ImmediateResult:
        body = []
        body = "".join(f"\n    %{var.address} =l alloc8 8" for _, var in scope.vars())
        body += "".join(stmt.ir for stmt in statements)
        return ImmediateResult(None, body)

    @staticmethod
    def function(scope: Scope, name: str, body: ImmediateResult) -> ImmediateResult:
        top = f"\nexport function l ${name}() {{\n@start"
        bottom = "\n}\n"
        return ImmediateResult(None, top + body.ir + bottom)

    @staticmethod
    def program(scope: Scope, main_func: ImmediateResult) -> ImmediateResult:
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
            + main_func.ir,
        )
