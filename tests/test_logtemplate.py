"""Tests for the in-repo Drain miner and the LogPai grouping metrics."""

from mivia_small_json import logtemplate as lt


def test_drain_groups_same_shape_messages():
    """Messages that differ only in a masked parameter share a template."""
    d = lt.Drain()
    g1, _ = d.add("request 12345 completed in 12 ms")
    g2, t2 = d.add("request 67890 completed in 34 ms")
    assert g1 == g2
    assert t2 == "request <*> completed in <*>"


def test_drain_separates_unmasked_word_at_tree_position():
    """The deeper default tree keys on leading tokens.

    With the measured default (depth=5, path_depth=3) a message that differs
    in an *unmasked* word at a tree position starts a new branch; parameter
    values (numbers, IPs, dotted tokens) are masked, so they do not. This is
    the trade-off that lifts LogHub GA from 0.7552 to 0.8149.
    """
    d = lt.Drain()
    g1, _ = d.add("user alice logged in")
    g2, _ = d.add("user bob logged in")
    assert g1 != g2


def test_drain_separates_different_token_counts():
    d = lt.Drain()
    g1, _ = d.add("short message here")
    g2, _ = d.add("a much longer message with many more tokens")
    assert g1 != g2


def test_drain_is_deterministic():
    msgs = ["a b c", "a b d", "x y z", "a b c"]

    def run():
        d = lt.Drain()
        return [d.add(m)[1] for m in msgs]

    assert run() == run()


def test_drain_empty_message():
    d = lt.Drain()
    assert d.add("") == (-1, "")


def test_grouping_metrics_perfect_partition():
    ga, fga = lt.grouping_metrics(["E1", "E1", "E2", "E2"],
                                  ["g0", "g0", "g1", "g1"])
    assert ga == 1.0
    assert fga == 1.0


def test_grouping_metrics_merged_group():
    # All four lines predicted as one group: GA 0, pairwise F1 0.5.
    ga, fga = lt.grouping_metrics(["E1", "E1", "E2", "E2"],
                                  ["g0", "g0", "g0", "g0"])
    assert ga == 0.0
    assert abs(fga - 0.5) < 1e-9


def test_parse_accuracy_and_template_f1():
    assert lt.parse_accuracy(["a <*> c", "a <*> c"], ["a <*> c", "q r s"]) == 0.5
    # An identical label list is a perfect template partition.
    assert lt.template_f1(["t1", "t1", "t2"], ["t1", "t1", "t2"]) == 1.0


def test_evaluate_returns_the_four_metrics():
    out = lt.evaluate(["E1", "E1"], ["g0", "g0"], ["t <*>", "t <*>"],
                      ["t <*>", "t <*>"])
    assert set(out) == {"GA", "PA", "FGA", "FTA"}
    assert out["GA"] == 1.0 and out["PA"] == 1.0 and out["FGA"] == 1.0


def test_preprocess_masks_numbers_and_hex():
    assert lt.preprocess("count 42 idle") == "count <*> idle"
    assert lt.preprocess("offset -7 x") == "offset <*> x"
    assert lt.preprocess("ptr 0xdeadBEEF set") == "ptr <*> set"
    assert lt.preprocess("hash a489c868f0c37da93b76227c91bb0390 set") == (
        "hash <*> set")


def test_preprocess_masks_ip_and_dotted_tokens():
    assert lt.preprocess("from 192.168.0.1 here") == "from <*> here"
    assert lt.preprocess("build 1.2.3.4 done") == "build <*> done"
    assert lt.preprocess("ratio 3.14 ok") == "ratio <*> ok"
    assert lt.preprocess("proxy.cse.cuhk.edu.hk:5070 open") == (
        "<*>:<*> open")


def test_preprocess_keeps_ordinary_words():
    msg = "user alice logged in from host"
    assert lt.preprocess(msg) == msg
    assert lt.preprocess("") == ""


def test_preprocess_is_deterministic():
    msg = "user bob 42 10.0.0.1 1.2.3"
    assert lt.preprocess(msg) == lt.preprocess(msg)


def test_drain_masks_parameter_tokens_before_matching():
    d = lt.Drain()
    g1, _ = d.add("read 1234 bytes from 10.0.0.1 at 1.2.3")
    g2, t2 = d.add("read 9876 bytes from 10.0.0.9 at 4.5.6")
    assert g1 == g2
    assert t2 == "read <*> from <*> at <*>"
