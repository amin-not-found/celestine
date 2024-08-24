#!/usr/bin/env python3

import sys
from pathlib import Path
from subprocess import run
from argparse import ArgumentParser, Namespace

from parse import Parser
from lexer import Lexer
from qbe import QBE

DEBUG = True


def compile_file(args: Namespace) -> Path:
    path = Path(args.file)
    ssa_file = path.parent.joinpath(f"{path.stem}.ssa")
    asm_file = path.parent.joinpath(f"{path.stem}.s")
    out_file = path.parent.joinpath(f"{path.stem}.out")

    lexer = Lexer.lex_file(path)
    parser = Parser(lexer)

    try:
        program = parser.parse()
    except StopIteration:
        if len(parser.diagnostics) < 1:
            print(f"{lexer.text.name}: Unexpected end of file", file=sys.stderr)
            sys.exit(1)

    if len(parser.diagnostics) > 0:
        for diag in parser.diagnostics:
            print(diag, file=sys.stderr)
        sys.exit(1)

    if args.print:
        print(program)

    gen = QBE
    with open(ssa_file, "w", encoding="UTF-8") as file:
        ir = program.to_ir(gen)
        file.writelines(line + "\n" for _, line in ir.results())

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
    out_path = compile_file(args)

    if args.run:
        run([out_path.absolute()], check=True)


if __name__ == "__main__":
    main()
