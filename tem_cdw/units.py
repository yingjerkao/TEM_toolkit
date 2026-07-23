"""Unit conversion between reciprocal-lattice units (a*) and radians.

a* = 2π/a where a is the real-space lattice constant. TEM experiments
report q in units of a*; the Alekseev code uses radians throughout.
This module exists so unit conversions happen only at the user boundary.
"""
import math

TWO_PI = 2.0 * math.pi


def q_from_astar(q_astar: float) -> float:
    """Convert q from a* units to radians (q_rad = 2π · q_astar)."""
    return TWO_PI * q_astar


def astar_from_q(q_rad: float) -> float:
    """Convert q from radians to a* units."""
    return q_rad / TWO_PI
