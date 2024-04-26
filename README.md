# Celestine
Celestine is my experimental programming language. I first started with the idea of improving C but it's getting a bit far away from just that.

As of now, Celestine is in early development and it's only supported on 64-bit linux.
Here's an example of a program written in Celestine:
```rust
function main(): i32 = {
    let ascii_zero = 48;
    let newline = 10;
    let n = 3;
    // print "n*2=" 
    putchar 110; // n
    putchar 42;  // *
    putchar 50;  // 2
    putchar 61;  // =
    // print value of n*2
    putchar n * 2 + ascii_zero;
    putchar newline;
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
3. Variable declaration(read about [variables](#variables))
4. Expression statement 

### Comments
Single lined comments are denoted by usage of `//`.
As of right now, there's no support other types of comments.

### Expressions
Statements accept expressions and as of right now only 64-bit integers are supported and every expression evaluates to an integer.

#### Operators

You can do these operations in an expression:
|Operator|Operation                                     |Precedence
|-------------|-----------------------------------------|----------
|`-`          | Negative number                         | 1
|`!`          | Logical not                             | 1
|`as`         | Type casting between numbers            | 2
|`*`, `/`, `%`| Multiplication, division, and remainder | 3
|`+`, `-`     | Addition and subtraction                | 4
|`<<`, `>>`   | Bitwise left and right shift            | 5
|`&`          | Bitwise AND                             | 6
|`^`          | Bitwise XOR                             | 7
|`\|`         | Bitwise OR                              | 8
|`==`, `!=`, `<`, `>`, `<=`, `>=`| Comparison           | 9
|`\|\|`       | Logical AND                             | 10
|`&&`         | Logical OR                              | 1  
|`=`          | Assignment                              | 12

Also you can group operations with parenthesis.

Example:
```ts
// prints 1 as 1+48 is the ASCII code for '1'
putchar ((4 + -1) * 2) / 3 - 1 + 48;
```
#### If Expression/Statement
Conditional execution of code is possible with `if`, `else if` and `else`. These keywords followed by a code block can be used both as expressions and statements. For now all if expressions return 0 but this will change in the future. Example:
```ts
if x % 2 == 0 {
    // Print "Even"
    putchar 69;
    putchar 118;
    putchar 101;
    putchar 110;
} else {
    // Print "Odd"
    putchar 79;
    putchar 100;
    putchar 100;
}
```
#### Loops
Only while loop is supported at the moment. While is an expression and return 0 as of now just like if. Example:
```ts
// prints a pyramid pattern
function main(): i32 = {
    let rows = 5;
    let star = 42;
    let newline = 10;

    let mut i = 1;
    while i <= rows {
        let mut j = 0;
        while j < i {
            putchar star;
            j = j + 1;
        }
        putchar newline;
        i = i + 1;
    }
    return 0;
}
```
### Variables
For now only integers are supported. Variables are immutable by default and you can make them mutable by using `mut` keyword. You can declare variables using `let` keyword:

```rust
let mut x;
let y = 1;
x = 42;
```

Also it's worth noting that variable assignment is an expression and returns the value assigned to variable. This behavior might change in the future.
```rust
let mut x;
let y = x = 10;
putchar y; // prints newline
```
### Data types
Currently only builtin primitive types are supported

#### Primitive types
Integer and floating points of 32 and 64 bit size are available. They're  called `i32`, `i64` , `f32` , `f64`. Integer and float literals have types of `i32` and `f32`. Also you can cast between these types using `as` keyword:
```rust
let a = 4;              // i32
let b = 1.6;            // f32
let c = (a as f32) * b; // f32
putchar c as i32 + 48;  
// prints '6' as c truncates to 6 in conversion and 48 is ascii code fore 0
```