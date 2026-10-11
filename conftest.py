"""Suite-wide guards that must hold for every test, in every order.

The autouse fixture pins the model-path constant. A test that assigns
``model.MODEL_PATH`` directly, instead of monkeypatching it, leaves the
module pointing at a directory pytest has deleted — every later test
that loads the repository artifact then silently reads the wrong file.
The previous guard was one test inside tests/test_weak_label_trust.py
and only worked because file order happened to run it after the
calibration tests; this fixture cannot be reordered out of the way.

This file was deleted as an empty tombstone in e647aa3; it now carries
real content again.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(session, config, items):
    """Record the suite's collected node ids for the inventory gate.

    addopts -q collapses plain ``pytest --collect-only -q`` output to
    per-file counts, so the gate reads the ids from the live session
    instead of parsing collection prose. Filtering applied later in this
    hook (-k, --lf) does not change what was collected.
    """
    config._collected_test_ids = sorted(item.nodeid for item in items)


@pytest.fixture(autouse=True)
def model_path_constant_is_left_alone():
    """Snapshot ``model.MODEL_PATH`` before each test; assert it after.

    The assert runs in this fixture's finalizer. This fixture does NOT
    request ``monkeypatch``: finalizers run in reverse setup order, so a
    per-test monkeypatch restores the attribute before the assert sees
    it, and correct monkeypatch usage passes. A plain assignment is
    never restored, so it fails here even when the test itself passed.
    """
    import mivia_small_json.model as model_mod

    snapshot = model_mod.MODEL_PATH
    yield
    assert model_mod.MODEL_PATH == snapshot, (
        f"model.MODEL_PATH leaked: {model_mod.MODEL_PATH!r} != "
        f"{snapshot!r}. A test rewrote the module attribute without "
        "monkeypatch (or restored it too late); the leaked path outlives "
        "the test and misroutes every later artifact load."
    )
