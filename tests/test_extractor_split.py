"""The extractor held-out split is a pure function of the system name.

Each extractor training script holds a system out iff sha256(name) % 4 == 0,
so adding a new log system never reshuffles the systems already assigned.
This pins that behavior and the agreement of the helper across the three
scripts. The scripts are not importable as modules, so each is loaded via
importlib (as tests/test_backfill.py does).
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "train_level_extractor.py",
    "train_component_extractor.py",
    "train_implicit_severity.py",
]

SYSTEMS = [
    "Android", "Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp",
    "Linux", "Mac", "OpenSSH", "OpenStack", "Proxifier", "Spark",
    "Thunderbird", "Windows", "Zookeeper",
]


def _load_script(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def splitters():
    return [(name, _load_script(name).held_out_split) for name in SCRIPTS]


def test_split_matches_sha256_mod_four(splitters):
    for name, held_out_split in splitters:
        for system in SYSTEMS:
            expected = int.from_bytes(
                hashlib.sha256(system.encode("utf-8")).digest(), "big"
            ) % 4 == 0
            assert held_out_split(system) is expected, (name, system)


def test_split_is_independent_of_how_systems_are_listed(splitters):
    """The bucket depends only on the name, never on listing order.

    A position-based split (every 4th of a sorted listing) would move a
    system's bucket when another system is added; a name-hashed one cannot.
    The property is that `held_out_split` reads only its argument, so the
    check is that no global state can change the answer between calls.
    """
    for name, held_out_split in splitters:
        first = {s: held_out_split(s) for s in SYSTEMS}
        second = {s: held_out_split(s) for s in SYSTEMS}
        assert first == second, name


def test_all_scripts_agree_on_the_split(splitters):
    reference = splitters[0][1]
    expected = [reference(s) for s in SYSTEMS]
    for name, held_out_split in splitters[1:]:
        assert [held_out_split(s) for s in SYSTEMS] == expected, name


def test_split_holds_out_a_fraction_of_the_known_systems(splitters):
    """The split is not degenerate: some but not all systems are held out."""
    for name, held_out_split in splitters:
        held = [s for s in SYSTEMS if held_out_split(s)]
        assert 0 < len(held) < len(SYSTEMS), (name, held)


SHIPPED = [
    "level_extractor.skops",
    "component_extractor.skops",
    "implicit_severity.skops",
]


def test_generalization_estimate_excludes_its_held_out_systems():
    """A reported held-out score must not come from a model that trained on it.

    Each shipped extractor records a ``<name>.trainmeta.json``. The
    ``generalization`` block names the systems the *estimating* probe trained
    on and the systems it was scored on; those two sets must be disjoint, or
    the number is a self-report. The shipped artifact itself is trained on
    every system (``shipped_train_systems``), so its own score on the held-out
    systems is a training fit and must not be reported as generalization.
    """
    import json

    held = {
        s for s in SYSTEMS
        if int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big") % 4 == 0
    }
    assert held, "the split is degenerate: no held-out systems"
    for artifact in SHIPPED:
        meta_path = ROOT / "models" / (artifact + ".trainmeta.json")
        assert meta_path.is_file(), f"{artifact} ships no trainmeta sidecar"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gen = meta["generalization"]
        train = set(gen["train_systems"])
        scored = set(gen["held_out_systems"])
        assert scored and scored <= held, (
            f"{artifact}: generalization scored on {sorted(scored)}, "
            f"not a subset of the held-out split"
        )
        assert not (train & scored), (
            f"{artifact}: generalization probe trained on the systems it scored "
            f"{sorted(train & scored)}: the held-out number is a self-report"
        )
