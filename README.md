# QuantisedEncoder

As part of this experiment I will attempt to create a quantised Protein Language Model trained *exclusively* on the fungi kingdom proteomes.

# TLDR Results

To be released...

# Authors
* Maciej Szczesny

# Methods
- Get the proteomes from [the fungi kingdom](https://www.uniprot.org/taxonomy/4751)
- Cluster at 40% identity (38333 to 16542 sequences, after thr 512 cutoff 10894 sequences)
![Length Distribution](https://github.com/skurl/QuantisedEncoder/blob/main/diagrams/length_distribution.png?raw=true)
- Train the model using BART-like architecture (based on my previous project)

To do's : 
- [ ] Batch Packing
- [ ] Quantisation aware training
- [ ] Post training quantisation
- [ ] Introduce Pre-trainig quantisation (QAT)
- [ ] Introduce Quantisation afterwards
- [ ] Compaere, contrast, BLOSUM62, ProteinGym analysis
- [ ] Write up results

# Results

# Diagrams

# Limitations

# Conclusion

![Mushroom](https://github.com/skurl/QuantisedEncoder/blob/main/diagrams/mushroom.png?raw=true)


