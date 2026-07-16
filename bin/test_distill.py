"""Self-check for distillation alignment + loss. Run: python bin/test_distill.py
No teacher download -- fakes teacher reps to test the geometry that matters."""
import torch
import torch.nn as nn

from core import Vocab, distill_loss, train_collate

vocab = Vocab.from_sequences(["ACDEFGHIK", "MKTAYW"])
x, y, attn, seqs = train_collate(["ACDEFGHIK", "MKTAYW"], vocab)   # masked batch + raw seqs preserved
assert seqs == ["ACDEFGHIK", "MKTAYW"], "raw sequences must survive collation for the teacher"

specials = torch.tensor([vocab.cls, vocab.eos, vocab.pad, vocab.mask, vocab.unk])
valid = attn & (y == -100) & ~torch.isin(x, specials)             # unmasked real residues
# every valid position is a real residue that was NOT masked (label -100) and not padding
assert valid.sum() > 0 and (x[valid].unsqueeze(-1) != specials).all()

B, L, Ds, Dt = x.shape[0], x.size(1), 16, 24
student = torch.randn(B, L, Ds)
proj = nn.Linear(Ds, Dt)

# identical (up to the projection) reps -> cosine 1 -> loss ~0; orthogonal-ish random -> loss ~1
same = distill_loss(student, proj, proj(student).detach(), valid).item()
diff = distill_loss(student, proj, torch.randn(B, L, Dt), valid).item()
assert same < 1e-5, f"aligned reps should give ~0 loss, got {same}"
assert 0.5 < diff < 1.5, f"unrelated reps should give ~1 loss, got {diff}"

# length mismatch (teacher padded shorter) must not crash -- min-length truncation handles it
distill_loss(student, proj, torch.randn(B, L - 1, Dt), valid[:, :L]).item()
print(f"distill self-check OK  (aligned {same:.2e}, unrelated {diff:.3f}, valid residues {int(valid.sum())})")
