import pytest
import math

from geometry.calculator import (
    rectangle_area,
    triangle_area,
    circle_area,
    trapezoid_area,
    delta_wing_area,
)

# =========================================================
# 1. RECTANGLE AREA TESTS (Formula: Length * Width)
# =========================================================

@pytest.mark.parametrize("length, width, expected", [
    (10,8,80),
    (10.5, 8.2, pytest.approx(86.1)),
    (2.5, 4, pytest.approx(10.0))
])

def test_rectangle_area_valid(length, width, expected):
    assert rectangle_area(length, width) == expected

# def test_rectangle_area_valid():
#     assert rectangle_area(10, 8) == 80

def test_rectangle_area_invalid():
    with pytest.raises(ValueError):
        rectangle_area(-10,8)

def test_rectangle_area_zero():
    with pytest.raises(ValueError):
        rectangle_area(0,7)


# =========================================================
# 1. TRIANGLE AREA TESTS (Formula: 0.5 * base * height)
# =========================================================

@pytest.mark.parametrize("height ,base, expected", [
    (10, 4, 20.0),                           
    (5.5, 2.0, pytest.approx(5.5)),          
    (7.2, 3, pytest.approx(10.8)),          
])
def test_triangle_area_valid(base, height, expected):
    assert triangle_area(base, height) == expected


# def test_triangle_area_valid():
#     assert triangle_area(10, 4) == 20

def test_triangle_area_invalid():
    with pytest.raises(ValueError):
        triangle_area(-10, 4)

def test_triangle_area_zero():
    with pytest.raises(ValueError):
        triangle_area(0, 4)

# =========================================================
# 2. CIRCLE AREA TESTS (Formula: pi * radius^2)
# =========================================================

@pytest.mark.parametrize("radius, expected", [
    (5, pytest.approx(78.539816)),           
    (2.5, pytest.approx(19.634954)),         
    (1, pytest.approx(math.pi)),             
])
def test_circle_area_valid(radius, expected):
    assert circle_area(radius) == expected

# def test_circle_area_valid():
#     assert circle_area(5) == pytest.approx(78.539816, rel=1e-5)


def test_circle_area_valid():
    assert circle_area(5) == pytest.approx(78.539816, rel=1e-5)


def test_circle_area_invalid():
    with pytest.raises(ValueError):
        circle_area(-5)


def test_circle_area_zero():
    with pytest.raises(ValueError):
        circle_area(0)


# =========================================================
# 3. TRAPEZOID AREA TESTS (Formula: 0.5 * (b1 + b2) * h)
# =========================================================

@pytest.mark.parametrize("base1, base2, height, expected", [
    (10, 6, 4, 32.0),                       
    (4.5, 3.5, 2.0, pytest.approx(8.0)),     
    (10, 5.5, 2, pytest.approx(15.5)),      
])
def test_trapezoid_area_valid(base1, base2, height, expected):
    assert trapezoid_area(base1, base2, height) == expected

# def test_trapezoid_area_valid():
#     assert trapezoid_area(10, 6, 4) == 32


def test_trapezoid_area_invalid():
    with pytest.raises(ValueError):
        trapezoid_area(-10, 6, 4)


def test_trapezoid_area_zero():
    with pytest.raises(ValueError):
        trapezoid_area(10, 0, 4)


# =========================================================
# 4. DELTA WING AREA TESTS (Formula: 0.5 * root_chord * wingspan)
# =========================================================


@pytest.mark.parametrize("span, root_chord,  expected", [
    (8, 4, 16.0),                            
    (2.5, 6.0, pytest.approx(7.5)),         
    (10.2, 5, pytest.approx(25.5)),         
])
def test_delta_wing_area_valid(span, root_chord, expected):
    assert delta_wing_area(span, root_chord) == expected

# def test_delta_wing_area_valid():
#     assert delta_wing_area(8, 4) == 16


def test_delta_wing_area_invalid():
    with pytest.raises(ValueError):
        delta_wing_area(-8, 4)


def test_delta_wing_area_zero():
    with pytest.raises(ValueError):
        delta_wing_area(8, 0)