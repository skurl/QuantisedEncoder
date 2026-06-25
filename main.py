'''
This is the layout I want to have for this project:


Import files from HuggingFace datasets, initially only the most representative subset


Embed, encode

Encoder model - same as before, but with quantisation introduced

Success calculated using Top-1 prediction accuracy % as well as perplexity.

Could do SAE analysis afterwards (maybe)

1. Fork, then delete. Copy super_runner.py into the new project and remove the Listeria-specific half: read_fasta, the FASTA glob, greedy_cluster, clusters_from_mmseqs, cluster_split. UniRef50 is pre-split, so all of that goes. Keep everything from masking downward unchanged.

2. Drop in the streaming loader. Replace the deleted data section with datasets.load_dataset("Synthyra/uniref50", streaming=True), and .take(100_000) for the first subset. This is the one real piece of new code.

3. Smoke-test on the subset before anything else. Tiny model, ~2k steps, single seed. You're only checking: does it run end-to-end, does loss fall, is perplexity in a sane range (~14–16 on a tiny undertrained model). Fast loop, cheap. Don't touch the full 60M until this is green.

4. Fix the config-drift bug now, while the code is fresh. Save the config inside the checkpoint (torch.save({"cfg": asdict(cfg), "model": sd}, ...)) and have analyze/quantize read architecture from it. This permanently kills the num_layers/length_cutoff class of bugs that cost you hours here.

5. Then scale, in this order: model up to the README's ESM2-8M config (d_model 320, vocab 33) → switch to a step budget (~50k steps, not epochs) → dropout=0 (now correct — you can't overfit 60M sequences) → checkpoint every N steps (cluster's flaky).

6. Benchmark, then quantise. Compare to the README's ESM2-8M UniRef50 row (ppl ~13), then run the int8 quant script as the capstone.

'''