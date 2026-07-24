import math
from tem_cdw.units import q_from_astar, astar_from_q, TWO_PI


def test_q_from_astar_zero():
    assert q_from_astar(0.0) == 0.0


def test_q_from_astar_quarter():
    assert q_from_astar(0.25) == math.pi / 2


def test_roundtrip():
    for x in [0.0, 0.15, 0.25, 0.5, 1.0]:
        assert astar_from_q(q_from_astar(x)) == x


def test_two_pi_constant():
    assert TWO_PI == 2.0 * math.pi
