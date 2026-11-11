import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from run_experiments import EVAL_SEEDS, check_seed_split, train_seed


def test_training_seeds_never_hit_test_seeds():
    for seed in (0, 1, 2, 3, 4):             # 300 episodes per run is fine; only training seed 9 would collide
        check_seed_split(seed, 300)


def test_overlap_is_caught():
    with pytest.raises(AssertionError):
        check_seed_split(9, 2)                 # 1000*9 + 1 = 9001 is a test seed


def test_formula():
    assert train_seed(2, 7) == 2007 and EVAL_SEEDS == [9001, 9002, 9003]