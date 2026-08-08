import math


"""Reusable geometry calculations for UAV-AERO ENGINE."""

def _validate_positive(value, name):
    """Validate that a geometry dimension is positive."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if value <=0:
        raise ValueError(f"{name} must be greater than zero.")

#---------------------------------------------------------------

def rectangle_area(length, width):
    _validate_positive(length, "length")
    _validate_positive(width, "width")

    return length * width
# try:
#     area = rectangle_area(2,"g")
#     print(area)
# except (ValueError, TypeError) as e:
#     print(e)

def triangle_area(height, base):
    _validate_positive(height, "height")
    _validate_positive(base, "base")
    return 0.5 * height * base
# try:
#     area = triangle_area(3,-6)
#     print(area)
# except (ValueError, TypeError) as e:
#     print(e)


def circle_area(radius):
    _validate_positive(radius, "radius")
    return math.pi * radius**2
# try:
#     area = circle_area("k")
#     print(area)
# except (ValueError, TypeError) as e:
#     print(e)


def trapezoid_area(base1, base2, height):
    _validate_positive(base1, "base1")
    _validate_positive(base2, "base2")
    _validate_positive(height, "height")
    return 0.5 * (base1 + base2) * height
# try:
#     area = trapezoid_area(15, -3, 38)
#     print(area)
# except (ValueError, TypeError) as e:
#     print(e)


def delta_wing_area(span, root_chord):
    """Calculate the planform area of a simple delta wing."""
    _validate_positive(span, "span")
    _validate_positive(root_chord, "root_chord")

    return 0.5 * span * root_chord