from os import PathLike
from typing import TextIO, Optional


class CompilerError(Exception):
    def __init__(
        self,
        msg: str,
        text: TextIO,
        offset: int,
        *args: object,
        filepath: Optional[PathLike] = None,
    ) -> None:
        (line, col) = CompilerError.get_location(text, offset)
        location = f"{filepath}:{line}:{col}" if filepath else f"code:{line}:{col}"
        super().__init__(f"{location} : error: {msg}", *args)

    @staticmethod
    def get_location(
        text: TextIO,
        offset: int,
    ):
        text.seek(0)

        content = text.read(offset)
        if not content:
            return (1, 1)

        sp = content.splitlines(keepends=True)
        return (len(sp), len(sp[-1]) + 1)  # We need column value counted from one


class EndOfTokens(CompilerError, StopIteration):
    def __init__(
        self, text: TextIO, offset: int, *args: object, filepath: PathLike | None = None
    ):
        super().__init__(
            "Unexpected end of tokens", text, offset, *args, filepath=filepath
        )


class VariableRedeclareError(Exception):
    pass
