# scGPT inference components

`src/vcell/scgpt_encoder.py` adapts the encoder and binning from
[bowang-lab/scGPT](https://github.com/bowang-lab/scGPT/tree/cebd6fae655b9c585a4807daa3ac31bb764f06b4).
The pretrained checkpoint is downloaded separately from the author repository.

MIT License

Copyright (c) 2022 suber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# scFoundation inference and vocabulary

The scFoundation components in `src/vcell/foundation_encoders.py` and
`assets/teacher_vocab/scfoundation_gene_index.tsv` are adapted from BioMap's
[scFoundation repository](https://github.com/biomap-research/scFoundation/tree/397631c495eddf9ad6644fc00c6ea8139e651245).
Copyright 2023 BioMap (Beijing) Intelligence Technology Limited. Apache-2.0;
the complete license is in `licenses/APACHE_2_0.txt`.
Changes: retain only the frozen cell encoder, use standard PyTorch, strictly
verify encoding weights, merge duplicate symbols before normalization.
Weights are downloaded separately from the pinned `genbio-ai/scFoundation`
redistribution. This is not a claim of byte identity with the original author's
download. Check the original model's terms before redistribution or other use.

# State Embedding

`src/vcell/state_encoder.py` is adapted from
[Arc Institute State](https://github.com/ArcInstitute/state/tree/9bbfe78a434a55205e4de834e1ea99f85f7a3add).
Attribution: Arc Institute and the State authors. This file is licensed under
**CC BY-NC-SA 4.0**, rather than the project's MIT license. The complete license
is `licenses/STATE_CODE_LICENSE.txt`. Changes are identified in its module header.

State SE-100M weights are downloaded separately from `arcinstitute/SE-100M`.
They are governed by the **Arc Research Institute State Model Non-Commercial
License** and acceptable-use policy, included in `licenses/`. Its definition of
derivative work expressly includes distilled models. Using Qwen as the student
does not remove upstream model restrictions. Final 421 exports include these
terms and the following citation:

Adduri, A. et al. (2025). *Predicting cellular responses to perturbation across
diverse contexts with State.* https://doi.org/10.1101/2025.06.26.661135

# Geneformer

Geneformer V1-10M weights, gene medians, Ensembl mapping and token dictionary are
downloaded separately from the pinned author repository
[ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer).
Apache-2.0. The V1 rank-value preprocessing follows the author's tokenizer;
this adapter uses the mean of final-layer gene representations and a custom
VCell response head. It is not the official in-silico perturbation algorithm.
Theodoris, C. V. et al. (2023). *Transfer learning enables predictions in network
biology*. Nature. https://doi.org/10.1038/s41586-023-06139-9

# Universal Cell Embedding (UCE)

`src/vcell/uce_model.py` is adapted from the MIT-licensed
[snap-stanford/UCE](https://github.com/snap-stanford/UCE/tree/9c416007be15ad6753dc84af4468c1dc10421ab9)
model definition. The complete notice is in `licenses/UCE-MIT.txt`. Changes:
remove global warning suppression and path mutation; preserve architecture and
parameter names. The separate human adapter merges duplicate raw-count symbols,
samples expressed genes, preserves chromosome token ordering and normalizes
protein tokens before frozen inference. The VCell response head is not a native
UCE perturbation decoder.

Rosen, Y. et al. (2026). *Universal cell embedding provides a foundation model
for cell biology*. Nature. https://doi.org/10.1038/s41586-026-10689-z

Weights and auxiliary files are downloaded separately from pinned
`minwoosun/uce-650m` and `minwoosun/uce-misc` redistributions. Checkpoint hashes
are verified; original Figshare byte identity is not asserted. The upstream
[Figshare model deposit](https://figshare.com/articles/dataset/24320806) lists
CC BY 4.0; the HF card has MIT metadata but CC-BY-NC-ND 4.0 in its body. These
inconsistent notices are recorded, not resolved or relicensed by this project.
Do not infer unrestricted model/derivative redistribution from our code license.
State-derived student terms described above continue to apply.

# Functional annotations

`assets/function/` contains GO Biological Process 2023 and Reactome 2022 gene-set
snapshots obtained from the [Enrichr service](https://maayanlab.cloud/Enrichr/).
Exact retrieval URLs, access date and SHA256 hashes are in `sources.json`.
Attribute the Gene Ontology Consortium, Reactome and Enrichr authors for these
annotations. They are third-party annotation data, not newly MIT-licensed data
authored by this project. No user's gene list or expression matrix was submitted
to the service; the public libraries were downloaded in full.

# Round 8 public gene knowledge

Round 8 retrieves human gene identifiers, names and summaries through
[MyGene.info](https://mygene.info/) and functional association edges through the
[STRING 12.0 API](https://version-12-0.string-db.org/). Gene identifiers are sent
to these public services; expression matrices and perturbation-response labels
are not. Request/response snapshots and content hashes stay in the user's data
directory, rather than being bundled with the repository. Attribute the original
annotation providers, MyGene.info and STRING when using those assets. Their data
retain their respective terms and are not relicensed under this code's MIT
license. STRING association confidence is not a causal effect size or a signed
regulatory coefficient. Frozen Qwen text representations retain the existing
Qwen model terms; State-derived distillation restrictions above still apply.

# Round 15 HGNC naming annotations

`configs/round15_hgnc_previous_symbols.json` contains a 21-gene subset of the
official [HGNC complete set](https://hgnc.genenames.org/download/), downloaded
from its public Google Cloud bucket. Source URL, SHA256, retrieval timestamp,
HGNC identifiers, report links and Approved/previous-symbol ownership are
retained in the snapshot. These are gene naming annotations, not expression
data or learned prediction weights. Attribute the HUGO Gene Nomenclature
Committee when using them; their upstream terms are not replaced by the
repository's MIT code license. The server reads the bundled snapshot offline.
