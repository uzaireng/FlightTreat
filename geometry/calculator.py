import math


"""Reusable geometry calculations for UAV-AERO ENGINE."""

def _validate_positive(value, name):
    """Validate that a geometry dimension is positive."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if value <=0:
        raise ValueError(f"{name} must be greater than zero.")

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Invalid input! Please enter a correct value.")

#---------------------------------------------------------------

def rectangle_area(length, width):
    _validate_positive(length, "length")
    _validate_positive(width, "width")

    return length * width

def triangle_area(height, base):
    _validate_positive(height, "height")
    _validate_positive(base, "base")

    return 0.5 * height * base

def circle_area(radius):
    _validate_positive(radius, "radius")

    return math.pi * radius**2

def trapezoid_area(base1, base2, height):
    _validate_positive(base1, "root chord")
    _validate_positive(base2, "tip chord")
    _validate_positive(height, "height")
    
    return 0.5 * (base1 + base2) * height



def delta_wing_area(span, root_chord):
    """Calculate the planform area of a simple delta wing."""
    _validate_positive(span, "span")
    _validate_positive(root_chord, "root_chord")

    return 0.5 * span * root_chord
def handle_rectangle_area():
    try:
        l = get_float_input("Enter length: ")
        w = get_float_input("Enter width: ")
        area = rectangle_area(l, w)
        print(f"Area of rectangle: {area}")
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")


def handle_triangle_area():
    try:
        h = get_float_input("Enter height: ")
        b = get_float_input("Enter base: ")
        area = triangle_area(h, b)
        print(f"Area of triangle: {area}")
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")


def handle_circle_area():
    try:
        r = get_float_input("Enter radius: ")
        area = circle_area(r)
        print(f"Area of circle: {area}")
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")


def handle_trapezoid_area():
    try:
        b1 = get_float_input("Enter root chord: ")
        b2 = get_float_input("Enter tip chord: ")
        h = get_float_input("Enter height: ")
        area = trapezoid_area(b1, b2, h)
        print(f"Area of trapezoid: {area}")
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")


def handle_delta_wing_area():
    try:
        s = get_float_input("Enter span: ")
        c = get_float_input("Enter root chord: ")
        area = delta_wing_area(s, c)
        print(f"Area of delta wing: {area}")
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
    

#---------------------------------------------------------------

GEOMETRY_MENU = {
    "1" : {"title" : "RECTANGLE AREA", "func" : handle_rectangle_area},
    "2" : {"title" : "TRIANGLE AREA", "func": handle_triangle_area},
    "3" : {"title" : "CIRCLE AREA", "func" : handle_circle_area},
    "4" : {"title" : "TRAPEZOID AREA", "func" : handle_trapezoid_area},
    "5" : {"title" : "DELTA WING AREA", "func" : handle_delta_wing_area}
}