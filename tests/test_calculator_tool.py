import pytest
from services.tools.calculator_tool import calculate


def test_basic_arithmetic():
    assert calculate("2 + 2") == "4"
    assert calculate("100 - 37") == "63"
    assert calculate("12 * 8") == "96"
    assert calculate("10 / 4") == "2.5"
    assert calculate("10 // 3") == "3"
    assert calculate("10 % 3") == "1"


def test_powers_and_exponents():
    assert calculate("2 ** 8") == "256"
    assert calculate("2 ^ 8") == "256"  # ^ replaced with **
    assert calculate("32 ** 5") == "33554432"


def test_math_functions_and_constants():
    assert calculate("sqrt(144)") == "12"
    assert calculate("abs(-42)") == "42"
    assert calculate("round(pi, 2)") == "3.14"
    assert calculate("floor(3.9)") == "3"
    assert calculate("ceil(3.1)") == "4"


def test_order_of_operations():
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("(2 + 3) * 4") == "20"


def test_unary_operators():
    assert calculate("-5 + 10") == "5"
    assert calculate("+7 - -3") == "10"


def test_division_by_zero():
    res = calculate("10 / 0")
    assert "Division by zero" in res


def test_empty_and_syntax_error():
    assert "cannot be empty" in calculate("   ")
    assert "Invalid mathematical syntax" in calculate("2 + * 3")


def test_security_disallowed_calls():
    # Disallow builtins or arbitrary code execution
    assert "Error:" in calculate("__import__('os').system('dir')")
    assert "Error:" in calculate("open('chats.db')")
    assert "Error:" in calculate("eval('2+2')")


def test_exponent_overflow_protection():
    res = calculate("2 ** 999999999")
    assert "Error:" in res
