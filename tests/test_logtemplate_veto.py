"""Constant-token merge veto for the Drain parser (quality-gap-closure Track B).

A rare line that disagrees with an established cluster on a constant-like
token (Windows client ``SPP`` vs ``WindowsUpdateAgent``, Proxifier ``SOCKS5``
vs ``HTTPS``) must start its own cluster: under GA one contaminant poisons
every line of the host group. Small clusters still merge freely so genuine
parameter variation wildcards normally.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mivia_small_json.logtemplate import Drain, _veto_pair  # noqa: E402


def _groups(messages):
    drain = Drain()
    return [drain.add(m)[0] for m in messages]


def test_veto_pair_constants():
    assert _veto_pair("SPP.", "WindowsUpdateAgent.") is True
    assert _veto_pair("SOCKS5", "HTTPS") is True
    assert _veto_pair("Bye", "disconnected") is True


def test_veto_pair_parameters():
    assert _veto_pair("Fri", "Sat") is False  # closed-class date words
    assert _veto_pair("user=root", "user=guest") is False  # key=value params
    assert _veto_pair("blk_1", "blk_2") is False  # shared-prefix params
    assert _veto_pair("eth0", "eth1") is False
    assert _veto_pair("<*>", "anything") is False  # masked positions
    assert _veto_pair("same", "same") is False


def test_rare_contaminant_starts_new_cluster():
    host = ("Session: 30546173_4261722401 initialized by "
            "client WindowsUpdateAgent.")
    rare = "Session: 11152_142416105 initialized by client SPP."
    gids = _groups([host] * 12 + [rare])
    assert len(set(gids)) == 2
    assert gids.count(gids[0]) == 12


def test_small_clusters_merge_freely():
    gids = _groups(["session closed for user cyrus"] * 3
                   + ["session closed for user news"] * 3)
    assert len(set(gids)) == 1


def test_date_words_merge_at_scale():
    base = "connection from 192.168.1.1 at {} Jun 12 10:00:00 2020"
    gids = _groups([base.format("Fri")] * 12 + [base.format("Sat")])
    assert len(set(gids)) == 1
