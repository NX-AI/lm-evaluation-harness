from lm_eval.tasks.leaderboard.math.utils import is_equiv


def test_is_equiv_uses_math_verify_parser():
    assert is_equiv(r"\frac{1}{2}", "0.5")
    assert not is_equiv("1", "2")
