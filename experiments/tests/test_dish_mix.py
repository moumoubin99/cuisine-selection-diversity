import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dish_mix_decomposition import terms  # noqa: E402


def _rows(spec):
    # spec: (dish, bin, n_kept, n_total)
    return [("p", d, b, i < kept) for d, b, kept, n in spec for i in range(n)]


def test_pure_composition_has_zero_within_term():
    # dish a always kept, dish b never; Q4 holds more b than Q1
    rows = _rows([("a", 0, 8, 8), ("b", 0, 0, 2), ("a", 3, 2, 2), ("b", 3, 0, 8)])
    t = terms(rows)
    assert abs(t["total"] - (0.2 - 0.8)) < 1e-12
    assert abs(t["within"]) < 1e-12
    assert abs(t["composition"] - t["total"]) < 1e-12


def test_pure_within_dish_has_zero_composition_term():
    # same dish mix in both quartiles; keep rate falls inside each dish
    rows = _rows([("a", 0, 5, 5), ("b", 0, 5, 5), ("a", 3, 0, 5), ("b", 3, 0, 5)])
    t = terms(rows)
    assert abs(t["composition"]) < 1e-12
    assert abs(t["within"] - (-1.0)) < 1e-12
