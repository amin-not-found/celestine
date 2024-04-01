# Celestine
Celestine is my experimental programming language. I first started with the idea of improving C but it's getting a bit far away from just that.

As of now, this is pretty much the only program supported and only on 64-bit linux:
```ts
// prints "Hi\n"
function main(): int = {
    putchar 72;  // H
    putchar 105; // i
    putchar 10;  // \n
    return 0;
}
```

## Dependencies
- [QBE](https://c9x.me/compile/)
- Python (tested with 3.11)
- A compiler for GAS style assembly (tested with GCC)

## Usage
```console
usage: main.py [-h] [-r] [-p] [-k] file

Celestine compiler.

positional arguments:
  file

options:
  -h, --help   show this help message and exit
  -r, --run    run executable output
  -p, --print  print generated AST
  -k, --keep   don't delete file from each compiling stage
```

## Design
Celestine tries to be a low level system programming language that has support for a subset of modern language features but doesn't stray much far from C.

A Celestine program consists of a main function which itself is made os statements. Every statement ends with a semicolon(`;`). There are two kinds of statements as of right now:
1. putchar which prints given a number prints it's corresponding ASCII character. It exists as a way to test the language and will be removed or replaced ved with a function in the future.
2. return which ends the function returning the given number.

### Comments
Single lined comments are denoted by usage of `//`.
As of right now, there's no support other types of comments.

### Expressions
Statements accept expressions and as of right now only 64-bit integers are supported and every expression evaluates to an integer.

#### Operators

You can do these operations in an expression:
|Operator|Operation         |Precedence
|--------|------------------|----------
|`+`     | Positive number  | 1
|`-`     | Negative number  | 1
|`!`     | Logical not      | 1
|`*`     | Multiplication   | 2
|`/`     | Division         | 2
|`+`     | Addition         | 3
|`-`     | Subtraction      | 3

Also you can group operations with parenthesis.

Example:
```ts
// prints 1 as 1+48 is the ASCII code for '1'
putchar ((4 + -1) * 2) / 3 - 1 + 48;
```

