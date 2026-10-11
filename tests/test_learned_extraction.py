"""Tests for learned log field extraction (component, level, and implicit severity)."""

from __future__ import annotations

import numpy as np

from mivia_small_json import extract_learned, pipeline


def test_learned_extractors_load_successfully():
    lvl_model = extract_learned.load_level_extractor()
    assert lvl_model is not None, "level extractor failed sha256-verified load"

    comp_model = extract_learned.load_component_extractor()
    assert comp_model is not None, "component extractor failed sha256-verified load"

    impl_model = extract_learned.load_implicit_severity_extractor()
    assert impl_model is not None, "implicit severity extractor failed sha256-verified load"


def test_component_extraction_on_syslog_and_daemons():
    comp_model = extract_learned.load_component_extractor()

    # Linux sshd with PID
    linux_line = (
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; "
        "logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4"
    )
    comp = extract_learned.extract_component(linux_line, comp_model)
    assert comp == "sshd"
    assert not comp.isdigit()

    # ftpd daemon
    ftpd_line = "Jun 14 15:16:02 combo ftpd[1234]: connection from 1.2.3.4"
    assert extract_learned.extract_component(ftpd_line, comp_model) == "ftpd"


def test_implicit_severity_inference():
    impl_model = extract_learned.load_implicit_severity_extractor()

    auth_fail = "authentication failure; logname= uid=0 euid=0 tty=NODEVssh"
    assert extract_learned.extract_implicit_severity(auth_fail, impl_model) == "ERROR"

    break_in = (
        "reverse mapping checking getaddrinfo for ns.attacker.com [1.2.3.4] "
        "failed - POSSIBLE BREAK-IN ATTEMPT!"
    )
    assert extract_learned.extract_implicit_severity(break_in, impl_model) == "WARN"

    session = "session closed for user cyrus"
    assert extract_learned.extract_implicit_severity(session, impl_model) == "INFO"


def test_extract_log_event_integration():
    lvl_model = extract_learned.load_level_extractor()
    comp_model = extract_learned.load_component_extractor()
    impl_model = extract_learned.load_implicit_severity_extractor()

    linux_line = (
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; "
        "logname= uid=0 euid=0"
    )
    obj = extract_learned.extract_log_event(
        linux_line,
        level_extractor=lvl_model,
        component_extractor=comp_model,
        implicit_severity_extractor=impl_model,
    )
    assert obj["type"] == "log_event"
    assert obj["level"] == "ERROR"
    assert obj["component"] == "sshd"
    assert "19939" not in obj["component"]


def test_pipeline_classify_syslog():
    raw = (
        "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; "
        "logname= uid=0 euid=0"
    )
    obj, meta = pipeline.classify(raw)
    assert meta["valid"] is True
    assert obj["type"] == "log_event"
    assert obj["level"] == "ERROR"
    assert obj["component"] == "sshd"


class _ColumnFlipModel:
    """Binary classifier whose positive class sits at column 0 (classes_=[1, 0])."""

    classes_ = [1, 0]

    def __init__(self, positive_tokens):
        self._positive = {t.lower() for t in positive_tokens}

    def predict_proba(self, feats):
        rows = []
        for feat_list in feats:
            toks = [f[4:] for f in feat_list if f.startswith("tok=")]
            hit = any(t in self._positive for t in toks)
            rows.append([0.9, 0.1] if hit else [0.1, 0.9])
        return np.array(rows)


def test_extract_level_reads_the_declared_positive_class():
    model = _ColumnFlipModel({"error"})
    assert extract_learned.extract_level("ERROR [auth]: boom", model) == "ERROR"


def test_extract_component_reads_the_declared_positive_class():
    model = _ColumnFlipModel({"sshd"})
    text = "Jun 14 15:16:01 combo sshd[19939]: authentication failure"
    assert extract_learned.extract_component(text, model) == "sshd"


class _MultiClassModel:
    """A model with no binary positive class: the extractor must refuse it."""

    classes_ = ["a", "b", "c"]

    def predict_proba(self, feats):
        return np.full((len(feats), 3), 1.0 / 3.0)


def test_incompatible_model_warns_and_returns_empty(capsys):
    model = _MultiClassModel()
    assert extract_learned.extract_level("ERROR boom", model) == ""
    assert extract_learned.extract_component("sshd[1]: boom", model) == ""
    err = capsys.readouterr().err
    assert "extract_level" in err and "extract_component" in err
