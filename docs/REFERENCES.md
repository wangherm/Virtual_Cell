# Methods and data references

These are the references supplied with the project, not a claim that external
models or datasets are bundled. Record actual dataset versions, model revisions,
licenses and download dates in each experiment's provenance.

1. [Arc 2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026).
2. [State implementation and license](https://github.com/ArcInstitute/state).
3. [State Replogle checkpoint candidate](https://huggingface.co/arcinstitute/ST-HVG-Replogle).
4. [GEARS implementation and cross-cell-type limitations](https://github.com/snap-stanford/GEARS).
5. [Replogle et al. author-released data](https://doi.org/10.25452/figshare.plus.20029387).
6. [HepG2/Jurkat data, GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) and the [associated paper](https://www.nature.com/articles/s41588-025-02169-3).
7. [Arc merged State-Replogle dataset](https://huggingface.co/datasets/arcinstitute/State-Replogle-Filtered).
8. [Deep Mutual Learning, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html).
9. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).
10. [Official cell-eval](https://github.com/ArcInstitute/cell-eval).
11. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), used only by the optional API controller.

This implementation uses MSE mutual learning on continuous expression predictions,
not the original classification KL objective. Its cross-student contrastive
objective is an InfoNCE-style experiment with same-input positive pairs, not the
complete supervised contrastive recipe from the cited paper. These citations
provide context; they do not establish this implementation's efficacy or novelty.

For deployment, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to choose a build compatible with the target server.
