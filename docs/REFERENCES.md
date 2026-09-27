# 原始方法与数据来源

生物学背景链接沿用 v0.1；OpenAI API 实现核查于 2026-09-26。实际实验记录具体 commit、checkpoint 和下载时间。

1. Arc 2026 Virtual Cell Challenge 官方说明：
   https://arcinstitute.org/news/virtual-cell-challenge-2026
2. State 官方实现与许可：
   https://github.com/ArcInstitute/state
3. State 遗传扰动权重候选：
   https://huggingface.co/arcinstitute/ST-HVG-Replogle
4. GEARS 官方实现及明确的 cross-cell-type 限制：
   https://github.com/snap-stanford/GEARS
5. Replogle et al. 2022 作者发布的处理后数据：
   https://doi.org/10.25452/figshare.plus.20029387
6. Nadig / HepG2 / Jurkat 数据：
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667
   论文：https://www.nature.com/articles/s41588-025-02169-3
7. Arc 合并的数据发布：
   https://huggingface.co/datasets/arcinstitute/State-Replogle-Filtered
8. Deep Mutual Learning（CVPR 2018）：
   https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html
9. Supervised Contrastive Learning：
   https://arxiv.org/abs/2004.11362
10. 官方 cell-eval：
    https://github.com/ArcInstitute/cell-eval
11. OpenAI 官方 Structured Outputs（Python Responses API / responses.parse）：
    https://developers.openai.com/api/docs/guides/structured-outputs

本版 mutual loss 是连续表达预测上的 MSE 互学，不是直接复制分类任务的 KL。
本版对比目标是跨学生、同输入正对的 InfoNCE 式实验，不是原论文完整的 supervised contrastive 配方。
所有外部方法引用均为方法背景，不构成对本包有效性或新颖性的证明。
