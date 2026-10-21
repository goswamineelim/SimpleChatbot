import ast
import math
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Backward compatibility with older Python AST
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Protect against excessive power computation that causes denial of service
        if op_type is ast.Pow:
            if isinstance(right, (int, float)) and right > 10000:
                raise OverflowError("Exponent too large (max allowed: 10000)")

        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Unknown variable or constant: '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are supported.")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Disallowed function: '{func_name}'")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the calculated numerical result.
    Supports operators (+, -, *, /, //, %, **), math functions (sqrt, sin, cos, tan, log, exp, abs, round, floor, ceil), and constants (pi, e).

    Args:
        expression: The mathematical expression string to evaluate, e.g. '32**5' or 'sqrt(144) + 15 * 3'.

    Returns:
        The evaluated numerical result as a string, or an error message if the expression is invalid.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return "Error: Expression cannot be empty."

    # Replace common symbols like ^ with ** for exponents
    clean_expr = clean_expr.replace("^", "**")

    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except OverflowError:
        return "Error: Number too large or computation overflowed."
    except SyntaxError as e:
        return f"Error: Invalid mathematical syntax ({e.msg})."
    except Exception as e:
        return f"Error: {str(e)}"
