import os, tempfile
import torch
from runner import Transformer, save_checkpoint, load_checkpoint


def test_roundtrip():
    arch = {"vocab_size": 12, "d_model": 32, "num_heads": 4, "num_layers": 2,
            "d_ff": 64, "dropout": 0.0, "num_classes": 8, "pad_idx": 0}
    m = Transformer(arch)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pth")
        save_checkpoint(p, m, ["<pad>"] + list("ACDEFGHIJK"), list("ACDEFGHI"))
        m2, vocab, classes = load_checkpoint(p)
    assert m2.arch == arch, "arch drifted on reload"
    assert classes == list("ACDEFGHI")
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b), "weights changed on reload"
    assert m2(torch.randint(0, 12, (2, 5))).shape == (2, 5, 8)
    print("checkpoint roundtrip ok")


if __name__ == "__main__":
    test_roundtrip()
