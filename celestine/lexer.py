from typing import TextIO, NamedTuple
from collections.abc import Iterator
from enum import Enum, auto
from os import PathLike


class TokenKind(Enum):
    # Single-character tokens
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    SEMICOLON = auto()
    COLON = auto()
    COMMA = auto()
    COMMENT = auto()
    ASSIGNMENT = auto()
    PLUS = auto()
    MINUS = auto()
    ASTERISK = auto()
    SLASH = auto()
    PERCENT = auto()
    BANG = auto()
    AMPERSAND = auto()
    V_BAR = auto()
    CARET = auto()
    GT = auto()
    LT = auto()
    # Two or more character tokens
    L_AND = auto()
    L_OR = auto()
    SHIFT_L = auto()
    SHIFT_R = auto()
    GE = auto()
    LE = auto()
    EQ = auto()
    NE = auto()
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    # Keywords
    PUTCHAR = auto()  # TODO : remove in future
    FUNCTION = auto()
    RETURN = auto()
    LET = auto()
    MUT = auto()
    AS = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    IDENTIFIER = auto()


class Token(NamedTuple):
    kind: TokenKind
    offset: int
    lexeme: str


class UnrecognizedToken(Exception):
    def __init__(self, text: str, *args: object) -> None:
        self.text = text
        super().__init__(*args)


class Lexer(Iterator):
    single_char_tokens = {
        "(": TokenKind.LEFT_PAREN,
        ")": TokenKind.RIGHT_PAREN,
        "{": TokenKind.LEFT_BRACE,
        "}": TokenKind.RIGHT_BRACE,
        ";": TokenKind.SEMICOLON,
        ":": TokenKind.COLON,
        ",": TokenKind.COMMA,
        "=": TokenKind.ASSIGNMENT,
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "*": TokenKind.ASTERISK,
        "/": TokenKind.SLASH,
        "%": TokenKind.PERCENT,
        "!": TokenKind.BANG,
        "&": TokenKind.AMPERSAND,
        "|": TokenKind.V_BAR,
        "^": TokenKind.CARET,
        "<": TokenKind.LT,
        ">": TokenKind.GT,
    }

    two_char_tokens: dict[TokenKind, dict[str, TokenKind]] = {
        TokenKind.SLASH: {"/": TokenKind.COMMENT},
        TokenKind.ASSIGNMENT: {"=": TokenKind.EQ},
        TokenKind.BANG: {"=": TokenKind.NE},
        TokenKind.LT: {
            "=": TokenKind.LE,
            "<": TokenKind.SHIFT_L,
        },
        TokenKind.GT: {
            "=": TokenKind.GE,
            ">": TokenKind.SHIFT_R,
        },
        TokenKind.AMPERSAND: {"&": TokenKind.L_AND},
        TokenKind.V_BAR: {"|": TokenKind.L_OR},
    }

    keywords = {
        "putchar": TokenKind.PUTCHAR,
        "function": TokenKind.FUNCTION,
        "return": TokenKind.RETURN,
        "let": TokenKind.LET,
        "mut": TokenKind.MUT,
        "as": TokenKind.AS,
        "if": TokenKind.IF,
        "else": TokenKind.ELSE,
        "while": TokenKind.WHILE,
    }

    def __init__(self, text: TextIO):
        self._position = 0
        self._text = text
        self._peeked = False
        self._peeked_token = None
        assert self._text.seekable()

    @staticmethod
    def lex_file(path: PathLike):
        # pylint: disable=consider-using-with
        file = open(path, encoding="UTF-8", newline="")
        return Lexer(file)

    def __iter__(self):
        return self

    def __next__(self) -> Token:
        if self._peeked:
            self._peeked = False
            return self._peeked_token

        char = self._forward_char()

        while char.isspace():
            char = self._forward_char()

        if not char:
            self._position -= 1
            raise StopIteration

        kind = self.single_char_tokens.get(char)

        # handle two or more character tokens
        two_char_candids = self.two_char_tokens.get(kind)
        if two_char_candids is not None:
            new_kind = two_char_candids.get(self._peek_char())
            if new_kind is not None:
                kind = new_kind
                self._forward_char()

        if kind == TokenKind.COMMENT:
            # A line comment
            while char and char != "\n":
                char = self._forward_char()
            return self.next()

        if kind is None:
            self._back_char()
            if char.isdigit() or char == ".":
                return self.parse_number()
            if char.isalpha() or char == "_":
                return self.parse_id()
            self._forward_char()
            raise UnrecognizedToken(char)

        return Token(kind, self._position, char)

    def __del__(self):
        self._text.close()

    @property
    def text(self):
        return self._text

    @property
    def offset(self):
        return self._position

    def _back_char(self):
        """Move back one char"""
        self._position -= 1
        self._text.seek(self._position)

    def _forward_char(self) -> str:
        """Move one char forward"""
        self._text.seek(self._position)
        self._position += 1
        return self._text.read(1)

    def _peek_char(self) -> str:
        """Look at next char without moving"""
        token = self._forward_char()
        self._back_char()
        return token

    def next(self):
        """Get next token"""
        return next(self)

    def parse_number(self) -> Token:
        pos = self._position

        lexeme = []
        is_float = False

        while (char := self._forward_char()).isdigit():
            lexeme.append(char)

        if char == ".":
            is_float = True
            lexeme.append(char)
            while (char := self._forward_char()).isdigit():
                lexeme.append(char)

        self._back_char()
        lexeme = "".join(lexeme)

        kind = TokenKind.FLOAT if is_float else TokenKind.INTEGER

        return Token(kind, pos, lexeme)

    def parse_id(self) -> Token:
        pos = self._position
        lexeme = []
        char = self._forward_char()
        while char.isalnum() or char == "_":
            lexeme.append(char)
            char = self._forward_char()
        self._back_char()

        lexeme = "".join(lexeme)
        kind = self.keywords.get(lexeme) or TokenKind.IDENTIFIER

        return Token(kind, pos, lexeme)
