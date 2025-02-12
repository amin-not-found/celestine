from typing import Type, Callable
from abc import abstractmethod, ABCMeta


from types_abc import *  # pylint: disable=wildcard-import,unused-wildcard-import
from lexer import TokenKind
from scope import Scope, ScopeType
from gen import GenBackend, GenResult, BlockIR, IfArm
from types_info import (
    BaseType,
    PrimitiveType,
    Pointer,
    NumericalType,
    IntegerType,
    I32,
    I64,
    F32,
    F64,
)


UnaryOperator = Callable[[Type[PrimitiveTypeABC], str, Scope], ImmediateResult]
BinaryOperator = Callable[
    [Type[PrimitiveTypeABC], str, str, Scope],
    ImmediateResult,
]


class QBEType(TypeABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def signature(self) -> str: ...


class QBEPrimitiveType(QBEType, PrimitiveTypeABC, metaclass=ABCMeta):
    unary_ops = {}
    binary_ops = {}

    @classmethod
    def unary_op_func(cls, op: TokenKind) -> UnaryOperator:
        return getattr(cls, (cls.unary_ops[op]).__name__)

    @classmethod
    def binary_op_func(cls, op: TokenKind) -> BinaryOperator:
        return getattr(cls, (cls.binary_ops[op]).__name__)


class QBENumerical(QBEPrimitiveType, NumericalABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def sign_signature(self) -> str: ...

    @classmethod
    def convert(cls, instruction: str, var: str, scope: Scope):
        res_var = scope.temp()
        return ImmediateResult(
            res_var, f"    %{res_var} ={cls.signature} {instruction} %{var}"
        )

    @classmethod
    def bin_op(cls, instruction: str, a: str, b: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(
            res, f"    %{res} ={cls.signature} {instruction} %{a}, %{b}"
        )

    @classmethod
    def logic_bin_op(cls, instruction: str, a: str, b: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(res, f"    %{res} =w {instruction} %{a}, %{b}")

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
    def eq(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"ceq{cls.signature}", a, b, scope)

    @classmethod
    def ne(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"cne{cls.signature}", a, b, scope)

    @classmethod
    def gt(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}gt{cls.signature}", a, b, scope)

    @classmethod
    def lt(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}lt{cls.signature}", a, b, scope)

    @classmethod
    def ge(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}ge{cls.signature}", a, b, scope)

    @classmethod
    def le(cls, a: str, b: str, scope: Scope):
        return cls.logic_bin_op(f"c{cls.sign_signature}le{cls.signature}", a, b, scope)

    unary_ops = {
        **QBEPrimitiveType.unary_ops,
    }

    binary_ops = {
        **QBEPrimitiveType.binary_ops,
        TokenKind.PLUS: add,
        TokenKind.MINUS: subtract,
        TokenKind.ASTERISK: multiply,
        TokenKind.GT: gt,
        TokenKind.LT: lt,
        TokenKind.GE: ge,
        TokenKind.LE: le,
        TokenKind.EQ: eq,
        TokenKind.NE: ne,
    }


class QBESignedNumerical(QBENumerical, metaclass=ABCMeta):

    @classmethod
    def negative(cls, a: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(res, f"    %{res} ={cls.signature} neg %{a}")

    @classmethod
    def divide(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("div", a, b, scope)

    unary_ops = {
        **QBENumerical.unary_ops,
        TokenKind.MINUS: negative,
    }

    binary_ops = {
        **QBENumerical.binary_ops,
        TokenKind.SLASH: divide,
    }


class QBEUnsignedNumerical(QBENumerical, metaclass=ABCMeta):
    @classmethod
    def divide(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("udiv", a, b, scope)

    binary_ops = {
        **QBENumerical.binary_ops,
        TokenKind.SLASH: divide,
    }


class QBEInteger(QBENumerical, IntegerABC, metaclass=ABCMeta):
    @classmethod
    def assign(cls, value: str, scope: Scope):
        var = scope.temp()
        return ImmediateResult(var, f"    %{var} ={cls.signature} copy {value}")

    @classmethod
    def logical_not(cls, a: str, scope: Scope):
        res = scope.temp()
        return ImmediateResult(res, f"    %{res} =l ceq{cls.signature} %{a}, 0")

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

    unary_ops = {
        **QBENumerical.unary_ops,
        TokenKind.BANG: logical_not,
    }
    binary_ops = {
        **QBENumerical.binary_ops,
        TokenKind.AMPERSAND: bit_and,
        TokenKind.V_BAR: bit_or,
        TokenKind.CARET: bit_xor,
        TokenKind.SHIFT_L: left_shift,
    }


class QBEFloat(QBESignedNumerical, FloatABC, metaclass=ABCMeta):
    sign_signature = ""

    @classmethod
    def assign(cls, value: str, scope: Scope):
        var = scope.temp()
        return ImmediateResult(
            var, f"    %{var} ={cls.signature} copy {cls.signature}_{value}"
        )


class QBESignedInteger(QBEInteger, QBESignedNumerical, metaclass=ABCMeta):
    sign_signature = "s"

    @classmethod
    def remainder(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("rem", a, b, scope)

    @classmethod
    def right_shift(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("sar", a, b, scope)

    unary_ops = {
        **QBEInteger.unary_ops,
        **QBESignedNumerical.unary_ops,
    }

    binary_ops = {
        **QBEInteger.binary_ops,
        **QBESignedNumerical.binary_ops,
        TokenKind.PERCENT: remainder,
        TokenKind.SHIFT_R: right_shift,
    }


class QBEUnsignedInteger(QBEInteger, metaclass=ABCMeta):
    sign_signature = "u"

    @classmethod
    def remainder(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("urem", a, b, scope)

    @classmethod
    def right_shift(cls, a: str, b: str, scope: Scope):
        return cls.bin_op("shr", a, b, scope)

    binary_ops = {
        **QBEInteger.binary_ops,
        TokenKind.PERCENT: remainder,
    }


class QBEi64(QBESignedInteger, I64ABC):
    signature = "l"

    @classmethod
    def from_i32(cls, var: str, scope: Scope):
        return cls.convert("extsw", var, scope)

    @classmethod
    def from_f32(cls, var: str, scope: Scope):
        return cls.convert("stosi", var, scope)

    @classmethod
    def from_f64(cls, var: str, scope: Scope):
        return cls.convert("dtosi", var, scope)


class QBEi32(QBESignedInteger, I32ABC):
    signature = "w"

    @classmethod
    def from_i64(cls, var: str, scope: Scope):
        return ImmediateResult(var, "")

    @classmethod
    def from_f32(cls, var: str, scope: Scope):
        return cls.convert("stosi", var, scope)

    @classmethod
    def from_f64(cls, var: str, scope: Scope):
        return cls.convert("dtosi", var, scope)


class QBEf32(QBEFloat, F32ABC):
    signature = "s"

    @classmethod
    def from_i32(cls, var: str, scope: Scope):
        return cls.convert("swtof", var, scope)

    @classmethod
    def from_i64(cls, var: str, scope: Scope):
        return cls.convert("sltof", var, scope)

    @classmethod
    def from_f64(cls, var: str, scope: Scope):
        return cls.convert("truncd", var, scope)


class QBEf64(QBEFloat, F64ABC):
    signature = "d"

    @classmethod
    def from_i32(cls, var: str, scope: Scope):
        return cls.convert("swtof", var, scope)

    @classmethod
    def from_i64(cls, var: str, scope: Scope):
        return cls.convert("sltof", var, scope)

    @classmethod
    def from_f32(cls, var: str, scope: Scope):
        return cls.convert("exts", var, scope)


class QBE(GenBackend):
    @staticmethod
    def get_type_impl(typ: BaseType):
        match typ:
            case I32():
                return QBEi32
            case I64():
                return QBEi64
            case F32():
                return QBEf32
            case F64():
                return QBEf64
            case Pointer():
                return QBEi64
            case _:
                raise NotImplementedError(f"No implementation for type '{typ}' on QBE")

    @staticmethod
    def literal(scope: Scope, typ: PrimitiveType, value: str) -> GenResult:
        typ = QBE.get_type_impl(typ)
        return GenResult.singular(typ.assign(value, scope), QBE)

    @staticmethod
    def var_assignment(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult:
        typ = QBE.get_type_impl(typ)
        var_signature = scope.var_signature(name)
        expr.append(
            expr.last().var,
            f"    store{typ.signature} %{expr.last().var}, {var_signature}",
        )
        return expr

    @staticmethod
    def deref_assignment(
        scope: Scope, ref: GenResult, typ: PrimitiveType, expr: GenResult
    ) -> GenResult:
        typ = QBE.get_type_impl(typ)
        expr_var = expr.last().var
        expr.concat(ref)  # Do the left side after the right side
        expr.append(
            expr.last().var,
            f"    store{typ.signature} %{expr_var}, %{ref.last().var}",
        )
        return expr

    @staticmethod
    def cast(
        scope: Scope, expr: GenResult, from_: NumericalType, to: NumericalType
    ) -> GenResult:
        typ: NumericalABC = QBE.get_type_impl(to)
        from_: NumericalABC = from_.abc_type

        var = expr.last().var
        if (from_) == I32ABC:
            res = typ.from_i32(var, scope)
        elif (from_) == I64ABC:
            res = typ.from_i64(var, scope)
        elif (from_) == F32ABC:
            res = typ.from_f32(var, scope)
        elif (from_) == F64ABC:
            res = typ.from_f64(var, scope)
        else:
            raise ValueError("Unreachable")
        expr.append_ir(res)
        return expr

    @staticmethod
    def logical_connective(
        scope: Scope,
        op: TokenKind,
        left: GenResult,
        right: GenResult,
        typ: PrimitiveType,
    ):
        res = scope.temp()
        resume_label = scope.label()
        end_label = scope.label()

        left.append(res, f"    %{res} =w copy %{left.last().var}")

        match op:
            case TokenKind.L_OR:
                left.append_str(
                    f"    jnz %{left.last().var}, @{end_label}, @{resume_label}"
                )
            case TokenKind.L_AND:
                left.append_str(
                    f"    jnz %{left.last().var}, @{resume_label}, @{end_label}"
                )
            case _:
                raise ValueError("Should be unreachable")

        left.append_str(f"@{resume_label}")
        left.concat(right)
        left.append(res, f"    %{res} =w copy %{right.last().var}")
        left.append(res, f"@{end_label}")

        return left

    @staticmethod
    def unary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        expr: GenResult,
    ):
        typ = QBE.get_type_impl(typ)
        func = typ.unary_op_func(op)
        expr.append_ir(func(expr.last().var, scope))
        return expr

    @staticmethod
    def address(scope: Scope, var: str) -> GenResult:
        var_state = scope.var_state(var)
        assert var_state is not None

        return GenResult.singular(ImmediateResult(f"{var_state.address}", ""), QBE)

    @staticmethod
    def dereference(
        scope: Scope,
        typ: Pointer,
        expr: GenResult,
    ) -> GenResult:

        var = scope.temp()
        typ = QBE.get_type_impl(typ)
        expr.append(
            var,
            f"    %{var} ={typ.signature} load{typ.signature} %{expr.last().var}",
        )

        return expr

    @staticmethod
    def binary_op(
        scope: Scope,
        op: TokenKind,
        typ: PrimitiveType,
        left: GenResult,
        right: GenResult,
    ):
        typ = QBE.get_type_impl(typ)
        func = typ.binary_op_func(op)
        left_var = left.last().var
        right_var = right.last().var

        left.concat(right)
        left.append_ir(func(left_var, right_var, scope))
        return left

    @staticmethod
    def if_expr(scope: Scope, arms: list[IfArm]) -> GenResult:
        assert len(arms) > 0

        end_label = scope.label()
        res = GenResult(QBE)

        for i, arm in enumerate(arms):
            arm_if = arm.body.start_label
            arm_start = scope.label()
            arm_else = (
                arms[i + 1].body.start_label if (i < len(arms) - 1) else end_label
            )
            arm_end = arm.body.end_label

            res.append_str(f"@{arm_if}")

            if arm.cond is None:
                # Else clause is the only arm without cond
                res.concat(arm.body.result)
                break

            res.concat(arm.cond)
            res.append(
                None,
                f"    jnz %{arm.cond.last().var}, @{arm_start}, @{arm_else}",
            )
            res.append_str(f"@{arm_start}")
            res.concat(arm.body.result)
            res.append_str(f"@{arm_end}")

            if not res.last().ir.strip().startswith("ret"):
                res.append_str(f"    jmp @{end_label}")

        res.append_str(f"@{end_label}")
        res_var = scope.temp()
        # TODO : actually use return value of if arms
        res.append(res_var, f"    %{res_var} =l copy 0")
        return res

    @staticmethod
    def while_expr(scope: Scope, cond: GenResult, body: BlockIR) -> GenResult:
        loop_label = scope.label()

        res = GenResult(QBE)
        res.append_str(f"@{body.start_label}")
        res.concat(cond)
        res.append_str(f"    jnz %{cond.last().var}, @{loop_label}, @{body.end_label}")
        res.append_str(f"@{loop_label}")
        res.concat(body.result)
        res.append_str(f"jmp @{body.start_label}")
        res.append(body.result.last().var, f"@{body.end_label}")

        return res

    @staticmethod
    def var(scope: Scope, name: str, typ: PrimitiveType) -> GenResult:
        typ = QBE.get_type_impl(typ)
        var_signature = scope.var_signature(name)

        var = scope.temp()
        return GenResult.singular(
            ImmediateResult(
                var,
                f"    %{var} ={typ.signature} load{typ.signature} {var_signature}",
            ),
            QBE,
        )

    @staticmethod
    def func_call(
        scope: Scope,
        name: str,
        params: list[tuple[GenResult, BaseType]],
        typ: PrimitiveType,
    ) -> GenResult:
        typ: QBEType = QBE.get_type_impl(typ)
        var = scope.temp()

        params_ir = []
        for param in params:
            param_typ = QBE.get_type_impl(param[1])
            params_ir.append(f"{param_typ.signature} %{param[0].last().var}")

        res = GenResult(QBE)
        for param, _ in params:
            res.concat(param)

        params_ir = ", ".join(params_ir)
        res.append(var, f"    %{var} ={typ.signature} call ${name}({params_ir})")
        return res

    @staticmethod
    def putchar(expr: GenResult, typ: IntegerType) -> GenResult:
        typ = QBE.get_type_impl(typ)
        expr.append_str(f"    call $putchar({typ.signature} %{expr.last().var})")
        return expr

    @staticmethod
    def return_stmt(expr: GenResult) -> GenResult:
        expr.append_str(f"    ret %{expr.last().var}")
        return expr

    @staticmethod
    def var_declare(
        scope: Scope, name: str, typ: PrimitiveType, expr: GenResult
    ) -> GenResult:
        return QBE.var_assignment(scope, name, typ, expr)

    @staticmethod
    def block(
        scope: Scope,
        start_label: str,
        end_label: str,
        statements: list[GenResult],
        gen_labels: bool,
    ) -> GenResult:
        res = GenResult(QBE)
        if gen_labels:
            res.append_str(f"@{start_label}")

        for _, var in scope.vars():
            if var.level == ScopeType.FUNC_ARG:
                continue
            res.append_str(f"    %{var.address} =l alloc8 {var.type.size}")

        for stmt in statements:
            res.concat(stmt)

        res_var = scope.temp()
        if not res.last().ir.strip().startswith("ret"):
            res.append(res_var, f"    %{res_var} =l copy 0")

        if gen_labels:
            res.append(res_var, f"@{end_label}")

        return res

    @staticmethod
    def function(
        scope: Scope,
        name: str,
        arguments: list[tuple[str, BaseType]],
        body: BlockIR,
        return_type: BaseType,
    ) -> GenResult:
        return_type = QBE.get_type_impl(return_type)
        prelude = GenResult(QBE)
        args = []

        for arg, typ in arguments:
            temp = scope.temp()
            var_state = scope.var_state(arg)

            prelude.append_str(f"    %{var_state.address} =l alloc8 {typ.size}")
            prelude.concat(
                QBE.var_assignment(
                    scope,
                    arg,
                    typ,
                    GenResult.singular(
                        ImmediateResult(temp, f"    # param {arg}"), QBE
                    ),
                )
            )

            typ = QBE.get_type_impl(typ)
            args.append(f"{typ.signature} %{temp}")

        res = GenResult(QBE)

        args = ", ".join(args)
        res.append_str(f"export function {return_type.signature} ${name}({args}) {{")
        res.append_str(f"@{body.start_label}")
        res.concat(prelude)
        res.concat(body.result)
        res.append_str(f"@{body.end_label}")
        # res.append_str(f"    ret %{body.result.last().var}")
        res.append_str("    ret")
        res.append_str("}\n")
        return res

    @staticmethod
    def program(scope: Scope, functions: list[GenResult]) -> GenResult:
        res = GenResult(QBE)
        res.append_str(
            """
function w $pushchar(w %c) { \n\
@start 
    %a =l alloc4 8 \n\
    storew %c, %a \n\
    %b =w call $write(w 1, w %a, w 1) \n\
    ret %b \n\
} \n"""
        )
        for func in functions:
            res.concat(func)
        return res
