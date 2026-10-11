"""Training determinism helper.

The shipped artifacts are pinned by sha256, but a hash pins a file, not a
build: two GPU fine-tunes with identical losses produced different artifacts.
`seed_everything` is what makes a re-run reproduce.
"""

import random

from mivia_small_json import determinism


def test_seed_everything_is_idempotent_for_a_seed():
    determinism.seed_everything(1234)
    first = [random.random() for _ in range(3)]
    determinism.seed_everything(1234)
    second = [random.random() for _ in range(3)]
    assert first == second


def test_different_seeds_give_different_streams():
    determinism.seed_everything(1)
    a = random.random()
    determinism.seed_everything(2)
    b = random.random()
    assert a != b


def test_seed_everything_returns_the_seed_and_never_raises():
    assert determinism.seed_everything(7) == 7
    # A non-int seed must not take the process down.
    try:
        determinism.seed_everything("not-a-seed")
    except Exception as exc:  # any raised type is fine; a crash is not
        raise AssertionError(f"non-int seed raised {type(exc).__name__}: {exc}")


def test_it_requests_the_cublas_workspace_config():
    determinism.seed_everything(42)
    import os

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
