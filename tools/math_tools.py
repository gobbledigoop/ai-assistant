# tools/math_tools.py

"""
Math Tools Module
-----------------
This module contains symbolic math tools powered by SymPy.
Each tool is defined as:
    - a pure Python function that performs the computation
    - a LangChain Tool wrapper that exposes it to the agent

Add new math tools here (derivatives, integrals, limits, solving, etc.)
"""

from sympy import symbols, sympify, diff, integrate, limit, solveset, S, latex
from langchain_core.tools import tool


# ---------------------------------------------------------
# Derivative Tool
# ---------------------------------------------------------

@tool
def derivative_tool(expression: str):
    """
    Compute the derivative of a mathematical expression with respect to x.
    """
    x = symbols("x")
    try:
        expression = expression.replace("^", "**")  # Handle caret for exponentiation
        expr = sympify(expression)
        result = diff(expr, x)
        return latex(result)
    except Exception as e:
        return f"Error computing derivative: {e}"



# ---------------------------------------------------------
# Integral Tool
# ---------------------------------------------------------

@tool
def integral_tool(expression: str):
    """
    Compute the indefinite integral of an expression with respect to x.
    """
    x = symbols("x")
    try:
        expression = expression.replace("^", "**")  # Handle caret for exponentiation
        expr = sympify(expression)
        result = integrate(expr, x)
        return latex(result)
    except Exception as e:
        return f"Error computing integral: {e}"


# ---------------------------------------------------------
# Limit Tool
# ---------------------------------------------------------

@tool
def limit_tool(input: str):
    """
    Compute the limit of an expression as x approaches a point.
    """
    try:
        x = symbols("x")
        expression, point = input.split(",")
        expr = sympify(expression.strip())
        pt = sympify(point.strip())
        result = limit(expr, x, pt)
        return latex(result)
    except Exception as e:
        return f"Error computing limit: {e}"


# ---------------------------------------------------------
# Equation Solver Tool
# ---------------------------------------------------------

@tool
def solve_tool(equation: str):
    """
    Solve an equation for x. Example: 'x^2 - 4 = 0'
    """
    x = symbols("x")
    try:
        equation = equation.replace("^", "**")  # Handle caret for exponentiation
        expr = sympify(equation)
        result = solveset(expr, x, domain=S.Complexes)
        return latex(result)
    except Exception as e:
        return f"Error solving equation: {e}"


# ---------------------------------------------------------
# Export list
# ---------------------------------------------------------

__all__ = [
    "derivative_tool",
    "integral_tool",
    "limit_tool",
    "solve_tool",
]
