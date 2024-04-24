from typing import Optional, NamedTuple
from abc import abstractmethod, ABCMeta

from lexer import TokenKind
from scope import Scope
from type import ImmediateResult, PrimitiveType, Integer, I64, I32


class IfArm(NamedTuple):
    cond: ImmediateResult
    body: ImmediateResult


def gen_method(func):
    return staticmethod(abstractmethod(func))


class GenBackend:
    # pylint: disable=unused-argument
    __metaclass__ = ABCMeta

    type_implementations: dict[PrimitiveType, PrimitiveType]

    @gen_method
    def int_literal(scope: Scope, integer_str: str) -> ImmediateResult: ...

    @gen_method
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: ImmediateResult,
    ) -> ImmediateResult: ...

    @gen_method
    def assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def logical_connective(
        scope: Scope, op: TokenKind, left: ImmediateResult, right: ImmediateResult
    ) -> ImmediateResult: ...

    @gen_method
    def binary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        left: ImmediateResult,
        right: ImmediateResult,
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
    def putchar(expr: ImmediateResult, typ: Integer) -> ImmediateResult: ...

    @gen_method
    def return_stmt(expr: ImmediateResult) -> ImmediateResult: ...

    @gen_method
    def var_declare(
        scope: Scope, name: str, typ: PrimitiveType, expr: ImmediateResult
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


class QBEInteger(Integer):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def signature(self) -> str: ...

    @classmethod
    def assign(cls, name: str, a: str, scope: Scope):
        var_signature = scope.var_signature(name)
        return ImmediateResult(
            None, f"\n    store{cls.signature} %{a}, {var_signature}"
        )

    @classmethod
    def bin_op(cls, instruction: str, a: str, b: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(
            res, f"\n    %{res} ={cls.signature} {instruction} %{a}, %{b}"
        )

    @classmethod
    def negative(cls, a: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(res, f"\n    %{res} ={cls.signature} neg %{a}")

    @classmethod
    def logical_not(cls, a: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(res, f"\n    %{res} =l ceq{cls.signature} %{a}, 0")

    @classmethod
    def add(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("add", a, b, scope)

    @classmethod
    def subtract(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("sub", a, b, scope)

    @classmethod
    def multiply(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("mul", a, b, scope)

    @classmethod
    def divide(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("div", a, b, scope)

    @classmethod
    def reminder(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("rem", a, b, scope)

    @classmethod
    def bit_and(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("and", a, b, scope)

    @classmethod
    def bit_or(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("or", a, b, scope)

    @classmethod
    def bit_xor(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("xor", a, b, scope)

    @classmethod
    def left_shift(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("shl", a, b, scope)

    @classmethod
    def right_shift(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("sar", a, b, scope)

    @classmethod
    def gt(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"csgt{cls.signature}", a, b, scope)

    @classmethod
    def lt(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"cslt{cls.signature}", a, b, scope)

    @classmethod
    def ge(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"csge{cls.signature}", a, b, scope)

    @classmethod
    def le(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"csle{cls.signature}", a, b, scope)

    @classmethod
    def eq(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"ceq{cls.signature}", a, b, scope)

    @classmethod
    def ne(cls, a: str, b: str, scope: Scope):
        return cls.bin_op(f"cne{cls.signature}", a, b, scope)

    @classmethod
    def logical_and(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        res = scope.temp()
        resume_label = scope.label()
        end_label = scope.label()

        ir = f"""{a.ir}
    %{res} =w copy %{a.var}
    jnz %{a.var}, @{resume_label}, @{end_label}
@{resume_label}{b.ir}
    %{res} =w copy %{b.var}
@{end_label}"""
        return ImmediateResult(res, ir)

    @classmethod
    def logical_or(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        res = scope.temp()
        resume_label = scope.label()
        end_label = scope.label()

        ir = f"""{a.ir}
    %{res} =w copy %{a.var}
    jnz %{a.var}, @{end_label}, @{resume_label}
@{resume_label}{b.ir}
    %{res} =w copy %{b.var}
@{end_label}"""
        return ImmediateResult(res, ir)


class QBEi64(QBEInteger, I64):
    signature = "l"


class QBEi32(QBEInteger, I32):
    signature = "w"


class QBE(GenBackend):
    type_implementations = {I64: QBEi64, I32: QBEi32}

    @staticmethod
    def int_literal(scope: Scope, integer_str: str) -> ImmediateResult:
        var_name = scope.temp()
        return ImmediateResult(var_name, f"\n    %{var_name} =l copy {integer_str}")

    @staticmethod
    def assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: ImmediateResult
    ) -> ImmediateResult:
        typ = QBE.type_implementations[typ]
        return ImmediateResult(expr.var, expr.ir + typ.assign(name, expr.var, scope).ir)

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
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: ImmediateResult,
    ):
        typ = QBE.type_implementations[typ]
        func = typ.unary_op_func(op)
        (var, ir) = func(expr.var, scope)
        ir = expr.ir + ir
        return ImmediateResult(var, ir)

    @staticmethod
    def binary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        left: ImmediateResult,
        right: ImmediateResult,
    ):
        typ = QBE.type_implementations[typ]
        func = typ.binary_op_func(op)
        (var, ir) = func(left.var, right.var, scope)
        ir = left.ir + right.ir + ir
        return ImmediateResult(var, ir)

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
    def putchar(expr: ImmediateResult, typ: Integer) -> ImmediateResult:
        typ = QBE.type_implementations[typ]
        ir = expr.ir + f"\n    call $putchar({typ.signature} %{expr.var})"
        return ImmediateResult(None, ir)

    @staticmethod
    def return_stmt(expr: ImmediateResult) -> ImmediateResult:
        return ImmediateResult(None, expr.ir + f"\n    ret %{expr.var}")

    @staticmethod
    def var_declare(
        scope: Scope, name: str, typ: PrimitiveType, expr: ImmediateResult
    ) -> ImmediateResult:
        return QBE.assignment(scope, name, typ, expr)

    @staticmethod
    def block(scope: Scope, statements: list[ImmediateResult]) -> ImmediateResult:
        body = "".join(
            f"\n    %{var.address} =l alloc{var.type.size} 8" for _, var in scope.vars()
        )
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
function w $pushchar(w %c) { \n\
@start 
    %a =l alloc4 8 \n\
    storew %c, %a \n\
    %b =w call $write(w 1, w %a, w 1) \n\
    ret %b \n\
} \n"""
            + main_func.ir,
        )
