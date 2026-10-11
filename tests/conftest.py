"""Suite-wide registration of the documented example schema.

``pull_request`` is registered through the public registry API exactly
like any project would, so every test — including the held-out gates —
exercises the four-schema world the committed corpus was generated in.

Every test also runs under a registry snapshot: registrations made
inside one test (or by ``load_active_router`` booting the shipped
world) are rolled back afterwards, so no test can change another's
schema world through global state.
"""

import pytest

from mivia_small_json import registry, synth

synth.register_example_schemas()


@pytest.fixture(autouse=True)
def _hermetic_registry():
    with registry.temporary_schemas():
        yield


@pytest.fixture
def corpus_world():
    """Enter the committed corpus world for this test (see bootstrap)."""
    from mivia_small_json import bootstrap

    with bootstrap.corpus_world():
        yield
