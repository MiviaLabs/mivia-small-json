"""The language-gated dual intent-rank head (2026-10-08, owner decision).

Two promoted heads behind ONE dispatch: English text routes the L6 head
(``intent-rank-int8.onnx`` — EN-only, measured 82.8-85 on English fresh
reserve), everything else routes the multilingual head
(``intent-rank-mul-int8.onnx`` — 62.2 on the multilang dev split where
the L6 scores 10.3). Dispatch is precision-first for the multilingual
side: non-English text hitting the EN head loses ~5x more than the
reverse, so AMBIGUOUS text routes multilingual. The shipped guard and
its select() contract are untouched — this head serves the
intent-selection surface (/v1/select ``head: "intent"``).
"""
from __future__ import annotations


from mivia_small_json import tool_match

B77_DESCS = {
    "card_arrival": "card arrival (card arrival)",
    "pending_transfer": "pending transfer (pending transfer)",
    "account_blocked": "account blocked (account blocked)",
}


def test_both_heads_promoted_and_verified():
    for name in ("intent-rank-int8.onnx", "intent-rank-mul-int8.onnx",
                 "intent-rank-tokenizer.json", "intent-rank-mul-tokenizer.json"):
        from pathlib import Path

        path = Path(tool_match.ROOT) / "models" / name
        assert path.is_file(), f"models/{name} missing"
        assert tool_match.integrity.verified(path), f"models/{name} sidecar"


def test_loaders_return_both_heads():
    en = tool_match.load_intent_rank()
    mul = tool_match.load_intent_rank_mul()
    assert en is not None and mul is not None
    assert en is not mul


def test_english_routes_the_l6_head():
    m = tool_match.select_intent("I am still waiting on my card, when does it arrive?",
                                 B77_DESCS)
    assert m["lang"] == "en"
    assert m["head"] == "en"
    assert m["choice"] in B77_DESCS


def test_non_english_routes_the_multilingual_head():
    for text in ("Wann kommt meine Karte endlich an?",
                 "Quand est-ce que ma carte arrive ?",
                 "カードがまだ届きませんが"):
        m = tool_match.select_intent(text, B77_DESCS)
        assert m["head"] == "mul", (text, m)
        assert m["choice"] in B77_DESCS


def test_ambiguous_ascii_text_prefers_multilingual():
    # Precision-first: "danke schon" without umlauts is German, not
    # English — a wrong EN route costs ~5x a wrong MUL route.
    m = tool_match.select_intent("danke schon, wann kommt meine karte",
                                 B77_DESCS)
    assert m["head"] == "mul", m


def test_detector_unit_contract():
    en = tool_match._prefers_multilingual
    assert en("where is my refund, I have been waiting for days") is False
    assert en("Mein Konto wurde gesperrt, warum?") is True
    assert en("¿Cuánto tarda en llegar mi tarjeta?") is True
    assert en("カードっていつ頃届きますか？") is True
    assert en("") is False
    assert en("12345 67890 !!!") is False


def test_select_head_parameter_validation():
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=object()))
    cands = list(B77_DESCS.values())
    resp = client.post("/v1/select", json={
        "text": "Wann kommt meine Karte an?",
        "candidates": cands, "head": "intent"})
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["choice"] in cands
    assert body["head"] == "mul"
    assert body["status"] in ("ok", "ambiguous")

    resp = client.post("/v1/select", json={
        "text": "x", "candidates": ["a"], "head": "nope"})
    assert resp.status_code == 422
