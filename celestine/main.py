#!/usr/bin/python3

import sys
from pathlib import Path
from subprocess import run
from argparse import ArgumentParser, Namespace

from lexer import Lexer
from errors import CompilerError
from ast_nodes import Program, Scope, ScopeType
from gen import QBE

DEBUG = True


def compile_file(args: Namespace) -> Path:
    path = Path(args.file)
    ssa_file = path.parent.joinpath(f"{path.stem}.ssa")
    asm_file = path.parent.joinpath(f"{path.stem}.s")
    out_file = path.parent.joinpath(f"{path.stem}.out")

    lexer = Lexer.lex_file(path)
    global_scope = Scope(ScopeType.GLOBAL)
    program = Program(lexer, global_scope)

    if args.print:
        print(program)

    gen = QBE()
    with open(ssa_file, "w", encoding="UTF-8") as f:
        f.write(program.to_ir(gen).ir)

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
        print(ce, file=sys.stderr)
        sys.exit(-1)

    if args.run:
        run([out_path.absolute()], check=True)


if __name__ == "__main__":
    main()
