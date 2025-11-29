#!/usr/bin/env python3
"""
Sample Math MCP Server
======================
A simple MCP server that provides math tools for testing.

Run with:
    python mcp_servers/math/server.py
"""

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Math Tools")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first.
    
    Args:
        a: First number
        b: Second number to subtract
        
    Returns:
        The difference (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        The quotient (a / b)
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise a number to a power.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        base raised to the power of exponent
    """
    return base ** exponent


@mcp.tool()
def sqrt(n: float) -> float:
    """Calculate the square root of a number.
    
    Args:
        n: The number to calculate the square root of
        
    Returns:
        The square root of n
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return n ** 0.5


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer.
    
    Args:
        n: A non-negative integer
        
    Returns:
        n! (n factorial)
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run(transport="stdio")

