"""Model tests: skops round-trip (save -> load -> identical predictions)."""

import skops.io as sio

from mivia_small_json import model, synth
from mivia_small_json.pipeline import load_router


def _tiny_examples():
    splits = synth.build_splits(12, 42)
    pairs = [(r["raw"], r["schema"]) for split in splits.values() for r in split]
    return [x for x, _ in pairs], [y for _, y in pairs]


def test_skops_roundtrip_same_predictions(tmp_path):
    X, y = _tiny_examples()
    pipe = model.build().fit(X, y)
    path = tmp_path / "router.skops"
    sio.dump(pipe, path)
    loaded = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    Xs = X[:10] + ["completely unseen random text line", "ERROR [db]: deadbeef"]
    assert (pipe.predict(Xs) == loaded.predict(Xs)).all()


def test_single_sha_verified_loader_no_artifact(tmp_path):
    """model.load_router was collapsed into pipeline.load_router; the
    one remaining loader degrades to None without an artifact."""
    assert not hasattr(model, "load_router")
    assert load_router(tmp_path / "missing.skops") is None


def test_artifact_and_sidecar_paths_are_single_sourced(tmp_path):
    """artifact_path/sidecar_path are the ONE answer to 'where is it'.

    Every construction of the artifact or sidecar path must go through
    these helpers; MODEL_PATH stays the repository constant and is
    defined through the helper, not beside it.
    """
    root = tmp_path / "somewhere"
    assert model.artifact_path(root) == root / "models" / "router.skops"
    assert model.artifact_path(str(root)) == root / "models" / "router.skops"
    assert model.artifact_path() == model.ROOT / "models" / "router.skops"
    assert model.sidecar_path(root) == root / "models" / "router.sha256"
    assert (model.sidecar_path(root).parent
            == model.artifact_path(root).parent)
    # The sidecar convention is <artifact-stem>.sha256 next to the
    # artifact, exactly what the loader reads.
    assert model.sidecar_path(root).stem == model.artifact_path(root).stem
    assert model.MODEL_PATH == model.artifact_path()


def test_lineage_manifest_hashes_every_training_corpus(tmp_path):
    (tmp_path / "data" / "synthetic").mkdir(parents=True)
    (tmp_path / "data" / "synthetic" / "train.jsonl").write_text(
        '{"raw": "a", "schema": "issue"}\n{"raw": "b", "schema": "commit"}\n',
        encoding="utf-8")
    manifest = model._data_manifest(tmp_path)
    assert [entry["path"] for entry in manifest] == ["data/synthetic/train.jsonl"]
    assert manifest[0]["rows"] == 2
    assert len(manifest[0]["sha256"]) == 64


def test_environment_and_trainmeta_path(tmp_path):
    assert set(model._environment()) >= {"python", "scikit-learn", "numpy",
                                         "scipy", "skops"}
    assert model.trainmeta_path(tmp_path) == (
        model.artifact_path(tmp_path).parent / "router.trainmeta.json")
