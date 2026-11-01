

import argparse
import ast
import cmath
import math
import operator
import re
from collections.abc import Callable, Mapping
from typing import TypeAlias

Number: TypeAlias = int | float | complex

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., object]] = {
    name: getattr(math, name)
    for name in (
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "comb",
        "copysign",
        "cos",
        "cosh",
        "degrees",
        "dist",
        "erf",
        "erfc",
        "exp",
        "expm1",
        "fabs",
        "factorial",
        "floor",
        "fmod",
        "frexp",
        "fsum",
        "gamma",
        "gcd",
        "hypot",
        "isclose",
        "isfinite",
        "isinf",
        "isnan",
        "isqrt",
        "lcm",
        "ldexp",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "modf",
        "perm",
        "radians",
        "remainder",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    )
}
_FUNCTIONS.update(
    {
        "abs": abs,
        "complex": complex,
        "max": max,
        "min": min,
        "round": round,
        "sum": sum,
        "ccos": cmath.cos,
        "cexp": cmath.exp,
        "clog": cmath.log,
        "csin": cmath.sin,
        "csqrt": cmath.sqrt,
        "ctan": cmath.tan,
    }
)
_CONSTANTS = {"e": math.e, "i": 1j, "pi": math.pi, "tau": math.tau}
_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)")


class CalculatorError(ValueError):
    pass


class _ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, names: Mapping[str, object]) -> None:
        self.names = names

    def visit_Expression(self, node: ast.Expression) -> object:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Number:
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float, complex)
        ):
            raise CalculatorError("Only numeric values are supported.")
        return node.value

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in self.names:
            return self.names[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id in _FUNCTIONS:
            raise CalculatorError("Functions must be called with parentheses.")
        raise CalculatorError(f"Unknown name: {node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> Number:
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise CalculatorError("That operator is not supported.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if not isinstance(left, (int, float, complex)) or not isinstance(
            right, (int, float, complex)
        ):
            raise CalculatorError("Arithmetic requires numeric values.")
        if isinstance(node.op, ast.Pow) and isinstance(right, int):
            left_bits = left.bit_length() if isinstance(left, int) else 0
            if abs(right) > 100_000 or left_bits * max(right, 0) > 1_000_000:
                raise CalculatorError("That power would produce an excessively large result.")
        return operation(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Number:
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise CalculatorError("That unary operator is not supported.")
        value = self.visit(node.operand)
        if not isinstance(value, (int, float, complex)):
            raise CalculatorError("Unary operators require numeric values.")
        return operation(value)

    def visit_List(self, node: ast.List) -> list[object]:
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[object, ...]:
        return tuple(self.visit(item) for item in node.elts)

    def visit_Call(self, node: ast.Call) -> object:
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only direct calls to calculator functions are allowed.")
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise CalculatorError(f"Unknown function: {node.func.id}")
        if node.keywords:
            raise CalculatorError("Named function arguments are not supported.")
        arguments = [self.visit(argument) for argument in node.args]
        try:
            return function(*arguments)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            raise CalculatorError(str(error)) from error

    def generic_visit(self, node: ast.AST) -> object:
        raise CalculatorError(f"Unsupported expression: {type(node).__name__}")


def evaluate(expression: str, names: Mapping[str, object] | None = None) -> object:
    
    normalized = expression.replace("^", "**").strip()
    if not normalized:
        raise CalculatorError("Enter an expression.")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as error:
        raise CalculatorError("Invalid expression.") from error
    try:
        return _ExpressionEvaluator(names or {}).visit(tree)
    except CalculatorError:
        raise
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise CalculatorError(str(error)) from error


def _show_help() -> None:
    print(
        "Enter an expression and press Enter. Examples: 2 + 3 * 4, "
        "sqrt(81), sin(pi / 2), 2^10\n"
        "Operators: +  -  *  /  //  %  ^ (or **)\n"
        "Functions include sin, cos, tan, log, log10, sqrt, factorial, "
        "gcd, comb, min, max, sum, round, csqrt, csin, and clog.\n"
        "Constants: pi, e, tau, i (imaginary unit). Assign values with name = expression; "
        "the previous result is ans.\n"
        "Commands: help, history, clear, quit"
    )


def run_repl() -> None:
    names: dict[str, object] = {}
    history: list[str] = []
    print("Scientific Calculator (type 'help' for help, 'quit' to exit)")
    while True:
        try:
            line = input("calc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        command = line.lower()
        if command in {"quit", "exit"}:
            return
        if command == "help":
            _show_help()
            continue
        if command == "history":
            for index, entry in enumerate(history, start=1):
                print(f"{index}: {entry}")
            continue
        if command == "clear":
            history.clear()
            print("History cleared.")
            continue

        assignment = _ASSIGNMENT.fullmatch(line)
        expression = line
        name: str | None = None
        if assignment:
            name, expression = assignment.groups()
            if name in _CONSTANTS or name in _FUNCTIONS:
                print(f"Error: '{name}' is a reserved name.")
                continue
        try:
            result = evaluate(expression, names)
        except CalculatorError as error:
            print(f"Error: {error}")
            continue
        if name is not None:
            names[name] = result
            print(f"{name} = {result}")
        else:
            names["ans"] = result
            print(result)
        history.append(line)


def run_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as error:
        raise RuntimeError(
            "The calculator window could not open. Tkinter and a desktop "
            "environment are required."
        ) from error
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "The calculator window could not open. Tkinter and a desktop "
            "environment are required."
        ) from error

    root.title("Scientific Calculator")
    root.geometry("620x780")
    root.minsize(520, 680)
    root.configure(background="#101827")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Calculator.TFrame", background="#101827")
    style.configure("Display.TFrame", background="#0b1220")
    style.configure(
        "Calculator.TLabel",
        background="#101827",
        foreground="#94a3b8",
        font=("Segoe UI", 10),
    )
    style.configure(
        "Heading.TLabel",
        background="#101827",
        foreground="#e2e8f0",
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Result.TLabel",
        background="#0b1220",
        foreground="#67e8f9",
        font=("Segoe UI", 11),
    )
    style.configure(
        "Display.TEntry",
        fieldbackground="#0b1220",
        foreground="#f8fafc",
        bordercolor="#334155",
        lightcolor="#334155",
        darkcolor="#334155",
        padding=(12, 10),
        font=("Segoe UI", 25, "bold"),
    )
    style.configure(
        "Calculator.TNotebook",
        background="#101827",
        borderwidth=0,
    )
    style.configure(
        "Calculator.TNotebook.Tab",
        background="#1e293b",
        foreground="#cbd5e1",
        padding=(16, 9),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Calculator.TNotebook.Tab",
        background=[("selected", "#334155")],
        foreground=[("selected", "#ffffff")],
    )
    style.configure(
        "Calculator.TButton",
        background="#1e293b",
        foreground="#f8fafc",
        borderwidth=0,
        padding=(5, 9),
        font=("Segoe UI", 12),
    )
    style.map(
        "Calculator.TButton",
        background=[("active", "#334155"), ("pressed", "#475569")],
    )
    style.configure(
        "Operator.TButton",
        background="#334155",
        foreground="#7dd3fc",
        borderwidth=0,
        padding=(5, 9),
        font=("Segoe UI", 12, "bold"),
    )
    style.map(
        "Operator.TButton",
        background=[("active", "#475569"), ("pressed", "#64748b")],
    )
    style.configure(
        "Equals.TButton",
        background="#0284c7",
        foreground="#ffffff",
        borderwidth=0,
        padding=(5, 9),
        font=("Segoe UI", 13, "bold"),
    )
    style.map(
        "Equals.TButton",
        background=[("active", "#0369a1"), ("pressed", "#075985")],
    )
    style.configure(
        "Clear.TButton",
        background="#3b2630",
        foreground="#fda4af",
        borderwidth=0,
        padding=(5, 9),
        font=("Segoe UI", 12, "bold"),
    )
    style.map(
        "Clear.TButton",
        background=[("active", "#57313d"), ("pressed", "#713949")],
    )

    container = ttk.Frame(root, padding=22, style="Calculator.TFrame")
    container.grid(row=0, column=0, sticky="nsew")
    container.columnconfigure(0, weight=1)
    container.rowconfigure(3, weight=1)

    ttk.Label(
        container,
        text="SCIENTIFIC CALCULATOR",
        style="Heading.TLabel",
    ).grid(row=0, column=0, sticky="w", pady=(0, 10))

    expression = tk.StringVar()
    display = ttk.Frame(container, padding=14, style="Display.TFrame")
    display.grid(row=1, column=0, sticky="ew", pady=(0, 14))
    display.columnconfigure(0, weight=1)
    entry = ttk.Entry(
        display,
        textvariable=expression,
        style="Display.TEntry",
        justify="right",
    )
    entry.grid(row=0, column=0, sticky="ew")

    result_text = tk.StringVar(value="Enter an expression or click the keys")
    ttk.Label(
        display,
        textvariable=result_text,
        style="Result.TLabel",
        anchor="e",
    ).grid(row=1, column=0, sticky="ew", pady=(9, 0))

    names: dict[str, object] = {}

    def calculate(_event: object = None) -> str:
        try:
            result = evaluate(expression.get(), names)
        except CalculatorError as error:
            result_text.set(f"Error: {error}")
        else:
            names["ans"] = result
            result_text.set(f"= {result}")
        return "break"

    def insert_text(text: str) -> None:
        if entry.selection_present():
            entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
        cursor = entry.index(tk.INSERT)
        entry.insert(cursor, text)
        entry.focus_set()

    def backspace() -> None:
        if entry.selection_present():
            entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
        else:
            cursor = entry.index(tk.INSERT)
            if cursor > 0:
                entry.delete(cursor - 1)
        entry.focus_set()

    def clear() -> None:
        expression.set("")
        result_text.set("")
        entry.focus_set()

    notebook = ttk.Notebook(container, style="Calculator.TNotebook")
    notebook.grid(row=3, column=0, sticky="nsew")

    keypads = {
        "Basic": [
            [("(", "("), (")", ")"), ("%", "%"), ("⌫", "backspace"), ("AC", "clear")],
            [("7", "7"), ("8", "8"), ("9", "9"), ("÷", "/"), ("×", "*")],
            [("4", "4"), ("5", "5"), ("6", "6"), ("−", "-"), ("+", "+")],
            [("1", "1"), ("2", "2"), ("3", "3"), ("^", "^"), ("=", "equals")],
            [("0", "0"), (".", "."), ("π", "pi"), ("e", "e"), ("i", "i")],
            [("Ans", "ans"), ("//", "//"), ("mod", "%"), ("x²", "^2"), ("√", "sqrt(")],
        ],
        "Scientific": [
            [("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("asin", "asin("), ("acos", "acos(")],
            [("atan", "atan("), ("sinh", "sinh("), ("cosh", "cosh("), ("tanh", "tanh("), ("exp", "exp(")],
            [("ln", "log("), ("log₁₀", "log10("), ("log₂", "log2("), ("abs", "abs("), ("n!", "factorial")],
            [("floor", "floor("), ("ceil", "ceil("), ("round", "round("), ("gcd", "gcd("), ("lcm", "lcm(")],
            [("min", "min("), ("max", "max("), ("sum", "sum("), ("comb", "comb("), ("perm", "perm(")],
            [("hypot", "hypot("), ("deg", "degrees("), ("rad", "radians("), ("C", "complex("), ("Ans", "ans")],
        ],
        "Complex": [
            [("i", "i"), ("C(", "complex("), ("√z", "csqrt("), ("sin z", "csin("), ("cos z", "ccos(")],
            [("tan z", "ctan("), ("exp z", "cexp("), ("log z", "clog("), ("abs", "abs("), ("Ans", "ans")],
            [("(", "("), (")", ")"), ("+", "+"), ("−", "-"), ("×", "*")],
            [("÷", "/"), ("^", "^"), ("%", "%"), ("⌫", "backspace"), ("AC", "clear")],
            [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("=", "equals")],
            [("4", "4"), ("5", "5"), ("6", "6"), ("7", "7"), ("8", "8")],
            [("9", "9"), (".", "."), ("pi", "pi"), ("e", "e"), ("i", "i")],
        ],
    }

    for tab_name, rows in keypads.items():
        tab = ttk.Frame(notebook, padding=8, style="Calculator.TFrame")
        for column in range(5):
            tab.columnconfigure(column, weight=1, uniform="calculator")
        for row_index, row in enumerate(rows):
            tab.rowconfigure(row_index, weight=1, uniform="calculator")
            for column_index, (label, action) in enumerate(row):
                if action == "equals":
                    command = calculate
                    button_style = "Equals.TButton"
                elif action == "backspace":
                    command = backspace
                    button_style = "Operator.TButton"
                elif action == "clear":
                    command = clear
                    button_style = "Clear.TButton"
                elif action == "factorial":
                    def factorial() -> None:
                        current = expression.get()
                        if current:
                            expression.set(f"factorial({current})")
                            entry.icursor(tk.END)
                        else:
                            insert_text("factorial(")

                    command = factorial
                    button_style = "Calculator.TButton"
                else:
                    command = lambda value=action: insert_text(value)
                    button_style = (
                        "Operator.TButton"
                        if action in {"+", "-", "*", "/", "//", "%", "^", "^2"}
                        else "Calculator.TButton"
                    )
                ttk.Button(
                    tab,
                    text=label,
                    command=command,
                    style=button_style,
                ).grid(
                    row=row_index,
                    column=column_index,
                    sticky="nsew",
                    padx=4,
                    pady=4,
                )
        notebook.add(tab, text=tab_name)

    entry.bind("<Return>", calculate)
    root.bind("<Escape>", lambda _event: clear())
    entry.focus_set()
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A safe scientific calculator.")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="use the interactive command-line calculator",
    )
    parser.add_argument("expression", nargs="*", help="expression to calculate")
    arguments = parser.parse_args(argv)
    if not arguments.expression:
        if arguments.cli:
            run_repl()
        else:
            run_gui()
        return 0
    expression = " ".join(arguments.expression)
    try:
        print(evaluate(expression))
    except CalculatorError as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())