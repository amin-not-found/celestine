from typing import Optional, NamedTuple
from abc import abstractmethod, ABCMeta

from lexer import TokenKind
from scope import Scope, ScopeType, Type
from type import (
    ImmediateResult,
    PrimitiveType,
    NumericalType,
    Integer,
    I32,
    I64,
    F32,
    F64,
)


class IfArm(NamedTuple):
    cond: ImmediateResult
    body: ImmediateResult


def gen_method(func):
    return staticmethod(abstractmethod(func))


class GenBackend(metaclass=ABCMeta):
    # pylint: disable=unused-argument
    type_implementations: dict[PrimitiveType, PrimitiveType]

    @gen_method
    def literal(scope: Scope, typ: PrimitiveType, value: str) -> ImmediateResult: ...

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
    def cast(
        scope: Scope, expr: ImmediateResult, from_: NumericalType, to: NumericalType
    ) -> ImmediateResult: ...

    @gen_method
    def logical_connective(
        scope: Scope,
        op: TokenKind,
        left: ImmediateResult,
        right: ImmediateResult,
        typ: PrimitiveType,
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
    def var(scope: Scope, name: str, typ: PrimitiveType) -> ImmediateResult: ...

    @gen_method
    def func_call(
        scope: Scope,
        name: str,
        params: list[tuple[ImmediateResult, Type]],
        typ: PrimitiveType,
    ) -> ImmediateResult: ...

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
    def function(
        scope: Scope,
        name: str,
        arguments: list[tuple[str, Type]],
        body: ImmediateResult,
        return_type: Type,
    ) -> ImmediateResult: ...

    @gen_method
    def program(scope: Scope, functions: list[ImmediateResult]) -> ImmediateResult: ...


class IfLabels(NamedTuple):
    if_: Optional[str]
    start: Optional[str]
    else_: Optional[str]


class QBEType(metaclass=ABCMeta):
    @property
    @abstractmethod
    def signature(self) -> str: ...


class QBEPrimitiveType(QBEType, PrimitiveType, metaclass=ABCMeta):
    pass


class QBENumerical(QBEPrimitiveType, NumericalType, metaclass=ABCMeta):

    @property
    @abstractmethod
    def sign_signature(self) -> str: ...

    @classmethod
    def convert(cls, instruction: str, value: ImmediateResult, scope: Scope):
        var = scope.temp()
        return ImmediateResult(
            var, value.ir + f"\n    %{var} ={cls.signature} {instruction} %{value.var}"
        )

    @classmethod
    def bin_op(
        cls, instruction: str, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ):
        res = scope.temp()
        ir = (
            a.ir
            + b.ir
            + f"\n    %{res} ={cls.signature} {instruction} %{a.var}, %{b.var}"
        )
        return ImmediateResult(res, ir)

    @classmethod
    def logic_bin_op(
        cls, instruction: str, a: ImmediateResult, b: ImmediateResult, scope: Scope
    ):
        res = scope.temp()
        ir = a.ir + b.ir + f"\n    %{res} =w {instruction} %{a.var}, %{b.var}"
        return ImmediateResult(res, ir)

    @classmethod
    def logical_not(cls, a: ImmediateResult, scope: Scope):
        res = scope.temp()
        return ImmediateResult(
            res, a.ir + f"\n    %{res} =l ceq{cls.signature} %{a.var}, 0"
        )

    @classmethod
    def add(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("add", a, b, scope)

    @classmethod
    def subtract(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("sub", a, b, scope)

    @classmethod
    def multiply(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("mul", a, b, scope)

    @classmethod
    def eq(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"ceq{cls.signature}", a, b, scope)

    @classmethod
    def ne(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"cne{cls.signature}", a, b, scope)

    @classmethod
    def gt(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}gt{cls.signature}", a, b, scope)

    @classmethod
    def lt(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}lt{cls.signature}", a, b, scope)

    @classmethod
    def ge(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}ge{cls.signature}", a, b, scope)

    @classmethod
    def le(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}le{cls.signature}", a, b, scope)


class QBESignedNumerical(QBENumerical, metaclass=ABCMeta):

    @classmethod
    def negative(cls, a: ImmediateResult, scope: Scope):
        res = scope.temp()
        return ImmediateResult(
            res, a.ir + f"\n    %{res} ={cls.signature} neg %{a.var}"
        )

    @classmethod
    def divide(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("div", a, b, scope)


class QBEInteger(QBENumerical, metaclass=ABCMeta):
    @classmethod
    def assign(cls, value: str, scope: Scope):
        var = scope.temp()
        return ImmediateResult(var, f"\n    %{var} ={cls.signature} copy {value}")

    @classmethod
    def bit_and(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("and", a, b, scope)

    @classmethod
    def bit_or(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("or", a, b, scope)

    @classmethod
    def bit_xor(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("xor", a, b, scope)

    @classmethod
    def left_shift(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("shl", a, b, scope)

    @classmethod
    def logical_and(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
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

    @classmethod
    def logical_or(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
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


class QBEFloat(QBESignedNumerical, metaclass=ABCMeta):
    sign_signature = ""

    @classmethod
    def assign(cls, value: str, scope: Scope):
        var = scope.temp()
        return ImmediateResult(
            var, f"\n    %{var} ={cls.signature} copy {cls.signature}_{value}"
        )


class QBESignedInteger(QBEInteger, QBESignedNumerical, metaclass=ABCMeta):
    sign_signature = "s"

    @classmethod
    def remainder(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("rem", a, b, scope)

    @classmethod
    def right_shift(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("sar", a, b, scope)


class QBEUnsignedInteger(QBEInteger, metaclass=ABCMeta):
    sign_signature = "u"

    @classmethod
    def remainder(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("urem", a, b, scope)

    @classmethod
    def right_shift(cls, a: ImmediateResult, b: ImmediateResult, scope: Scope):
        return cls.bin_op("shr", a, b, scope)


class QBEi64(QBESignedInteger, I64):
    signature = "l"

    @classmethod
    def from_i32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("extsw", value, scope)

    @classmethod
    def from_f32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("stosi", value, scope)

    @classmethod
    def from_f64(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("dtosi", value, scope)


class QBEi32(QBESignedInteger, I32):
    signature = "w"

    @classmethod
    def from_i64(cls, value: ImmediateResult, scope: Scope):
        return value

    @classmethod
    def from_f32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("stosi", value, scope)

    @classmethod
    def from_f64(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("dtosi", value, scope)


class QBEf32(QBEFloat, F32):
    signature = "s"

    @classmethod
    def from_i32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("swtof", value, scope)

    @classmethod
    def from_i64(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("sltof", value, scope)

    @classmethod
    def from_f64(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("truncd", value, scope)


class QBEf64(QBEFloat, F64):
    signature = "d"

    @classmethod
    def from_i32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("swtof", value, scope)

    @classmethod
    def from_i64(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("sltof", value, scope)

    @classmethod
    def from_f32(cls, value: ImmediateResult, scope: Scope):
        return cls.convert("exts", value, scope)


class QBE(GenBackend):
    type_implementations: dict[PrimitiveType, QBEPrimitiveType] = {
        I64: QBEi64,
        I32: QBEi32,
        F32: QBEf32,
        F64: QBEf64,
    }

    @staticmethod
    def literal(scope: Scope, typ: PrimitiveType, value: str) -> ImmediateResult:
        typ = QBE.type_implementations[typ]
        return typ.assign(value, scope)

    @staticmethod
    def assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: ImmediateResult
    ) -> ImmediateResult:
        typ = QBE.type_implementations[typ]
        var_signature = scope.var_signature(name)
        return ImmediateResult(
            expr.var,
            expr.ir + f"\n    store{typ.signature} %{expr.var}, {var_signature}",
        )

    @staticmethod
    def cast(
        scope: Scope, expr: ImmediateResult, from_: NumericalType, to: NumericalType
    ) -> ImmediateResult:
        typ: NumericalType = QBE.type_implementations[to]
        if (from_) == I32:
            return typ.from_i32(expr, scope)
        if (from_) == I64:
            return typ.from_i64(expr, scope)
        if (from_) == F32:
            return typ.from_f32(expr, scope)
        if (from_) == F64:
            return typ.from_f64(expr, scope)
        raise ValueError("Unreachable")

    @staticmethod
    def logical_connective(
        scope: Scope,
        op: TokenKind,
        left: ImmediateResult,
        right: ImmediateResult,
        typ: PrimitiveType,
    ):
        typ = QBE.type_implementations[typ]
        func = typ.binary_op_func(op)
        return func(left, right, scope)

    @staticmethod
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: ImmediateResult,
    ):
        typ = QBE.type_implementations[typ]
        func = typ.unary_op_func(op)
        return func(expr, scope)

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
        return func(left, right, scope)

    @staticmethod
    def if_expr(scope: Scope, arms: list[IfArm]) -> ImmediateResult:
        assert len(arms) > 0

        arm_labels: list[IfLabels] = []
        end_label = scope.label()

        else_label = scope.label()
        for arm in arms:
            arm_labels.append(IfLabels(else_label, scope.label(), scope.label()))
            else_label = arm_labels[-1].else_

        ir: str = []

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
            if not ir[-1].split("\n")[-1].strip().startswith("ret"):
                # TODO : Handle this case better
                ir.append(f"\n    jmp @{end_label}")
        else:
            ir.append(f"\n@{else_label}")

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
    def var(scope: Scope, name: str, typ: PrimitiveType) -> ImmediateResult:
        typ = QBE.type_implementations[typ]
        var_signature = scope.var_signature(name)

        var = scope.temp()
        return ImmediateResult(
            var, f"\n    %{var} ={typ.signature} load{typ.signature} {var_signature}"
        )

    @staticmethod
    def func_call(
        scope: Scope,
        name: str,
        params: list[tuple[ImmediateResult, Type]],
        typ: PrimitiveType,
    ) -> ImmediateResult:
        typ: QBEType = QBE.type_implementations[typ]
        var = scope.temp()

        params_ir = []
        for param in params:
            param_typ = QBE.type_implementations[param[1]]
            params_ir.append(f"{param_typ.signature} %{param[0].var}")

        params_ir = "(" + ", ".join(params_ir) + ")"
        ir = "".join(param[0].ir for param in params)

        return ImmediateResult(
            var, ir + f"\n    %{var} ={typ.signature} call ${name}{params_ir}"
        )

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
            f"\n    %{var.address} =l alloc{var.type.size} 8"
            for _, var in scope.vars()
            if var.level != ScopeType.FUNC_ARG
        )
        body += "".join(stmt.ir for stmt in statements)
        return ImmediateResult(None, body)

    @staticmethod
    def function(
        scope: Scope,
        name: str,
        arguments: list[tuple[str, Type]],
        body: ImmediateResult,
        return_type: Type,
    ) -> ImmediateResult:
        return_type = QBE.type_implementations[return_type]
        arg_assigns = []
        args = []

        for arg, typ in arguments:
            temp = scope.temp()
            var_state = scope.var_state(arg)

            arg_assigns.append(f"\n    %{var_state.address} =l alloc{typ.size} 8")
            arg_assigns.append(
                QBE.assignment(scope, arg, typ, ImmediateResult(temp, "")).ir
            )

            typ = QBE.type_implementations[typ]
            args.append(f"{typ.signature} %{temp}")

        args = ", ".join(args)
        arg_assigns = "".join(arg_assigns)

        top = f"\nexport function {return_type.signature} ${name}({args}) {{\n@start"
        body = body.ir
        bottom = "\n    \n}\n"

        if not body.split("\n")[-1].strip().startswith("ret"):
            # TODO : handle this situation better
            body = body + "\nret"

        return ImmediateResult(None, top + arg_assigns + body + bottom)

    @staticmethod
    def program(scope: Scope, functions: list[ImmediateResult]) -> ImmediateResult:
        functions = "".join(f.ir for f in functions)
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
            + functions,
        )
