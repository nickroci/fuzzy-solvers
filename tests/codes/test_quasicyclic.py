from __future__ import annotations

import pytest

from codes.linear import check_code, minimum_distance
from codes.quasicyclic import QuasiCyclic, polynomial_from_degrees, rotate

# codetables.de publishes [60,20,17] as a quasi-cyclic code of degree 3 with
# these generating polynomials (S. Weijs, M.Sc. thesis, Eindhoven, 1997).
PUBLISHED_60_20_17 = (
    polynomial_from_degrees((0,)),
    polynomial_from_degrees((14, 12, 9, 8, 7, 5, 4, 2, 0)),
    polynomial_from_degrees((17, 16, 15, 13, 12, 9, 8, 6, 5, 3, 1)),
)


def test_rotation_is_multiplication_modulo_x_to_the_block_minus_one() -> None:
    assert rotate(0b0001, 1, 4) == 0b0010
    assert rotate(0b1000, 1, 4) == 0b0001
    assert rotate(0b1010, 0, 4) == 0b1010
    assert rotate(0b0001, 4, 4) == 0b0001


def test_rejects_a_non_positive_block() -> None:
    with pytest.raises(ValueError, match="block length must be positive"):
        QuasiCyclic(block=0, polynomials=(1,))


def test_rejects_an_empty_generator_list() -> None:
    with pytest.raises(ValueError, match="at least one generator"):
        QuasiCyclic(block=4, polynomials=())


def test_rejects_a_polynomial_of_too_high_degree() -> None:
    with pytest.raises(ValueError, match="degree at least 4"):
        QuasiCyclic(block=4, polynomials=(0b10000,))


def test_shape_follows_from_index_and_block() -> None:
    qc = QuasiCyclic(block=7, polynomials=(1, 3, 5))
    assert qc.index == 3
    assert qc.length == 21
    assert qc.code().length == 21
    assert "QC index 3" in qc.describe()


def test_a_single_block_gives_a_cyclic_code() -> None:
    qc = QuasiCyclic(block=7, polynomials=(0b0001011,))
    assert qc.code().length == 7


def test_it_reproduces_the_published_60_20_17_code() -> None:
    """The construction must rebuild a real table entry from its polynomials."""
    code = QuasiCyclic(block=20, polynomials=PUBLISHED_60_20_17).code()
    assert code.length == 60
    assert code.dimension == 20
    assert minimum_distance(code) == 17
    assert check_code(code, 20, 17).accepted


def test_polynomial_from_degrees_sets_the_named_terms() -> None:
    assert polynomial_from_degrees((0, 3)) == 0b1001
    assert polynomial_from_degrees(()) == 0
