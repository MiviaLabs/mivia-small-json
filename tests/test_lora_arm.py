"""LoRA arm unit tests (CPU-only, fast; no GPU, no network).

Pin the four properties the folded export and the pre-registered arm
depend on (reports/exp-severity-lora.md, scripts/train_severity_lora.py):

(a) a fresh LoRALinear is the identity (B = 0 -> wrapped == frozen base);
(b) fold_lora reproduces the wrapper output after a real optimizer step
    perturbs A/B (the fold W0 + (alpha/r) B A is exact, not approximate);
(c) the built model freezes the whole backbone and trains only LoRA A/B
    plus the ordinal latent head and cutpoints — verified on the graph,
    not just on flags (backward must not reach frozen weights);
(d) an exported fp32->int8 artifact of the tiny model loads through the
    shipped loader (sha256 sidecar verified) and returns K-1 cumulative
    logits for a string.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# torch/transformers are training-extra deps; importorskip keeps the
# torch-free runtime install green (house pattern, tests/test_fillability_head.py).
torch = pytest.importorskip("torch")


HIDDEN, LAYERS, HEADS = 32, 2, 4
VOCAB = 30522  # matches models/router-tokenizer.json (bert-base-uncased)


def _tiny_encoder():
    from transformers import BertConfig, BertModel

    cfg = BertConfig(vocab_size=VOCAB, hidden_size=HIDDEN,
                     num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                     intermediate_size=64)
    return BertModel(cfg)


def test_lora_init_is_identity():
    from torch import nn

    import train_severity_lora as tsl

    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    with torch.no_grad():
        base.weight.normal_()
        base.bias.normal_()
    wrapped = tsl.LoRALinear(base)
    wrapped.eval()
    x = torch.randn(8, 16)
    with torch.no_grad():
        # B = 0 and dropout is the identity in eval mode: exact equality,
        # the wrapper must not change the frozen function at init.
        assert torch.equal(wrapped(x), base(x))
    # able to fail: any nonzero B breaks the identity
    with torch.no_grad():
        wrapped.lora_B.add_(0.1)
    with torch.no_grad():
        assert not torch.equal(wrapped(x), base(x))


def test_fold_equals_wrapper_after_training_step():
    from torch import nn

    import train_severity_lora as tsl

    class Box(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1)
            self.fc = tsl.LoRALinear(nn.Linear(16, 12))

        def forward(self, x):
            return self.fc(x)

    box = Box()
    x = torch.randn(32, 16)
    target = torch.randn(32, 12)
    box.eval()
    with torch.no_grad():
        y_init = box(x)  # identity at init (B = 0)
    opt = torch.optim.SGD([box.fc.lora_A, box.fc.lora_B], lr=0.1)
    box.train()
    for _ in range(3):
        loss = ((box(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # the fake training step must actually move B off zero AND change the
    # function, else the fold comparison below would be trivially true
    assert float(box.fc.lora_B.detach().abs().sum()) > 0.0
    box.eval()
    with torch.no_grad():
        y_wrapped = box(x)
    assert not torch.allclose(y_init, y_wrapped, atol=1e-4, rtol=1e-4)
    assert tsl.fold_lora(box) == 1
    assert not isinstance(box.fc, tsl.LoRALinear)
    with torch.no_grad():
        y_folded = box(x)
    assert torch.allclose(y_wrapped, y_folded, atol=1e-4, rtol=1e-4)


def test_backbone_frozen_lora_trainable_on_graph():
    import train_severity_lora as tsl

    torch.manual_seed(2)
    net = tsl.build_ordinal_lora(_tiny_encoder(), k=4, pool="cls")
    assert net.n_lora == 2 * LAYERS  # q + v per attention layer

    # flags: everything in the encoder is frozen except lora_A/lora_B
    for n, p in net.encoder.named_parameters():
        assert p.requires_grad == n.endswith(("lora_A", "lora_B")), n
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    lora = {n for n in trainable if n.endswith(("lora_A", "lora_B"))}
    head = {n for n in trainable
            if n.startswith(("latent", "cut0", "deltas"))}
    assert len(lora) == 2 * net.n_lora
    assert head == {"latent.weight", "latent.bias", "cut0", "deltas"}
    assert trainable == lora | head

    # graph: backward must reach the adapters and the head, never the
    # frozen backbone weights
    ids = torch.randint(0, VOCAB, (2, 10))
    mask = torch.ones_like(ids)
    net(ids, mask).sum().backward()
    # BertModel keeps the stack at encoder.layer.{i}.attention.self.{q,v}
    q0 = net.encoder.get_submodule("encoder.layer.0.attention.self.query")
    assert isinstance(q0, tsl.LoRALinear)
    assert q0.lora_A.grad is not None
    assert q0.lora_B.grad is not None
    assert q0.base.weight.grad is None
    assert net.encoder.embeddings.word_embeddings.weight.grad is None
    assert net.latent.weight.grad is not None
    assert net.deltas.grad is not None


def test_exported_int8_artifact_loads_and_scores(tmp_path):
    import torch.nn.functional as F
    from onnxruntime.quantization import QuantType, quantize_dynamic

    import train_severity_lora as tsl
    from mivia_small_json.tool_match import load_severity_ordinal

    torch.manual_seed(3)
    net = tsl.build_ordinal_lora(_tiny_encoder(), k=4, pool="cls")
    # one real CORAL step so the adapters are non-trivial before folding
    ids = torch.randint(0, VOCAB, (4, 12))
    mask = torch.ones_like(ids)
    golds = torch.tensor([0, 1, 2, 3])
    targets = (golds.unsqueeze(1) > torch.arange(3)).float()
    opt = torch.optim.SGD(
        [p for p in net.parameters() if p.requires_grad], lr=0.5)
    loss = F.binary_cross_entropy_with_logits(net(ids, mask), targets)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert float(net.encoder.get_submodule(
        "encoder.layer.0.attention.self.query").lora_B.detach().abs()
        .sum()) > 0.0

    net.eval()
    assert tsl.fold_lora(net) == 2 * LAYERS

    fp32 = tmp_path / "tiny-lora.onnx"
    int8 = tmp_path / "tiny-lora-int8.onnx"
    torch.onnx.export(
        net, (ids, mask), fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["cumulative_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "cumulative_logits": {0: "batch"}},
        opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    (tmp_path / (int8.name + ".sha256")).write_text(
        hashlib.sha256(int8.read_bytes()).hexdigest() + "\n",
        encoding="utf-8")

    head = load_severity_ordinal(int8)  # sha-verified, CPU onnxruntime
    assert head is not None, "int8 artifact failed the shipped loader"
    logits = head.logits("payment gateway returned a timeout after 30s")
    assert len(logits) == 3  # K-1 cumulative logits, same contract as shipped
    _anchor, _score, tiers = head.grade(
        "payment gateway returned a timeout after 30s", with_set=True)
    assert len(tiers) == 1 and 0 <= tiers[0] <= 3


def test_lora_scale_semantics_are_pinned():
    """Adversarial-review finding (2026-10-07): identity-at-init and
    fold-equality are invariant to a uniformly wrong scale (alpha/rank
    applied twice, or not at all) — those tests cannot see it. This one
    pins the SCALE semantics directly: with non-zero A/B, the wrapper's
    delta over the frozen base must equal exactly scale * x @ A^T @ B^T
    with scale = alpha / rank for the module's own alpha/rank values.
    """
    from torch import nn

    import train_severity_lora as tsl

    torch.manual_seed(1)
    for rank, alpha in ((4, 8.0), (8, 16.0), (4, 32.0)):
        base = nn.Linear(10, 6)
        wrapped = tsl.LoRALinear(base, rank=rank, alpha=alpha)
        with torch.no_grad():
            wrapped.lora_A.normal_()
            wrapped.lora_B.normal_()
        wrapped.eval()  # dropout off
        assert wrapped.scale == alpha / rank
        x = torch.randn(5, 10)
        with torch.no_grad():
            delta = wrapped(x) - base(x)
            manual = (alpha / rank) * (x @ wrapped.lora_A.T @ wrapped.lora_B.T)
        assert torch.allclose(delta, manual, atol=1e-6), (rank, alpha)
        # And the scale is not silently 1.0 or applied twice:
        with torch.no_grad():
            wrong_once = x @ wrapped.lora_A.T @ wrapped.lora_B.T
            wrong_twice = 2 * manual
        assert not torch.allclose(delta, wrong_once, atol=1e-3)
        assert not torch.allclose(delta, wrong_twice, atol=1e-3)
