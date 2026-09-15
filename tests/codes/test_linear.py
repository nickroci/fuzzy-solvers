from __future__ import annotations

import pytest

from codes.linear import (
    LinearCode,
    check_code,
    dual_code,
    hamming_code,
    minimum_distance,
    reduced_basis,
)


def test_rejects_a_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        LinearCode(length=0, rows=())


def test_rejects_a_row_outside_the_length() -> None:
    with pytest.raises(ValueError, match="outside a code of length"):
        LinearCode(length=3, rows=(0b1000,))


def test_dependent_rows_do_not_raise_the_dimension() -> None:
    code = LinearCode(length=4, rows=(0b1100, 0b0110, 0b1010))
    assert code.dimension == 2


def test_the_basis_is_reduced() -> None:
    """Each pivot column must be clear in every other row."""
    basis = reduced_basis((0b1100, 0b0110, 0b1010))
    pivots = [row.bit_length() - 1 for row in basis]
    for index, row in enumerate(basis):
        for other, pivot in enumerate(pivots):
            if other != index:
                assert not row >> pivot & 1


def test_a_trivial_code_has_no_nonzero_word() -> None:
    assert minimum_distance(LinearCode(length=5, rows=())) == 0
    assert minimum_distance(LinearCode(length=5, rows=(0,))) == 0


def test_the_repetition_code_has_distance_equal_to_its_length() -> None:
    assert minimum_distance(LinearCode(length=5, rows=(0b11111,))) == 5


@pytest.mark.parametrize("parity", [2, 3, 4, 5])
def test_hamming_codes_match_the_literature(parity: int) -> None:
    """Hamming codes are [2^r - 1, 2^r - 1 - r, 3]; the referee must agree."""
    code = hamming_code(parity)
    assert code.length == (1 << parity) - 1
    assert code.dimension == (1 << parity) - 1 - parity
    assert minimum_distance(code) == 3


def test_hamming_rejects_a_degenerate_redundancy() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        hamming_code(1)


def test_the_dual_annihilates_the_code() -> None:
    code = hamming_code(3)
    dual = dual_code(code)
    assert dual.dimension + code.dimension == code.length
    for row in code.basis():
        for check in dual.basis():
            assert (row & check).bit_count() % 2 == 0


def test_the_floor_short_circuits_the_enumeration() -> None:
    code = LinearCode(length=8, rows=(0b00000011, 0b11111100))
    assert minimum_distance(code, floor=5) < 5


def test_the_referee_accepts_a_code_meeting_its_claim() -> None:
    check = check_code(hamming_code(3), 4, 3)
    assert check.accepted
    assert check.diagnostics == ()
    assert "verified [7,4,3]" in check.render()


def test_the_referee_accepts_a_code_beating_its_claim() -> None:
    """A claim is a lower bound on what was found, so exceeding it is fine."""
    assert check_code(hamming_code(3), 4, 2).accepted


def test_the_referee_rejects_a_wrong_dimension() -> None:
    check = check_code(hamming_code(3), 5, 3)
    assert not check.accepted
    assert "dimension is 4" in check.render()


def test_the_referee_rejects_an_overstated_distance() -> None:
    check = check_code(hamming_code(3), 4, 4)
    assert not check.accepted
    assert "below the claimed" in check.render()
