# 第三章：研究方法

本章把第二章的研究问题转化为具体实验。答案质量和操作性失败检测被分别评价，失败终点在评价检测器之前固定（Carlini 等，2026）。

## 研究设计

本研究属于定量、回顾性和临床前评价。Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B 和 MedGemma-4B-it 在每个数据集上进行一次零样本推理。每个样本的一个 greedy answer 与参考答案比较，三个 sampled answers 用于构建黑盒失败分数。由于研究问题关注两个维度是否一致，论文不建立综合的“性能与可靠性”分数。三个模型在规模、训练、结构和医学专门化方面均不同，因此实验描述的是已发布检查点在受控接口下的行为，不能把差异因果归于单一组件。

## 模型与配置

三个检查点代表不同的指令跟随 VLM 家族，各自保留其支持的 processor 和聊天模板，但在同一数据集内接收完全相同的内容。四比特加载用于降低推理显存，并不作为实验变量；任何模型都没有在 PathVQA、VQA-RAD 或 ProstateMM-CHIMERA 上微调。

| 模型 | 参数规模 | 模型家族与结构 |
|---|---:|---|
| Qwen2.5-VL-3B-Instruct | 约 3B | 视觉编码器、多模态投影和指令微调语言骨干组成的 Qwen2.5-VL 模型（Bai 等，2023） |
| LLaVA-1.5-7B | 约 7B | CLIP 视觉编码器通过可学习投影连接到 7B 语言模型（Liu 等，2023） |
| MedGemma-4B-it | 约 4B | 面向生物医学图像与文本任务进行指令微调的 Gemma 医疗多模态模型（Sellergren 等，2025） |

## 评价框架

每个样本包含图像 $x_i$、问题 $q_i$ 和参考答案 $y_i$，ProstateMM-CHIMERA 还包含临床背景 $z_i$。同一数据集内的 prompt、测试样本、答案规则、指标实现、失败阈值和检测器设置在模型间固定。greedy answer 用于答案质量和操作性标签，sampled answers 用于检测分数，最后比较两种模型排名。正式论文中的方法图展示了这一分工以及 AUROC、AUPRC 和选择性分析；图中的接受或人工复核分支仅为示意，不是经过临床验证的策略。

## 推理协议

prompt 不包含训练样本，也不更新模型权重；二元问题要求只输出 yes 或 no，开放问题要求简短医学短语。主答案和三个采样答案分别为 $\hat y_i^{(0)}=\operatorname{GreedyDecode}[p_\theta(y\mid x_i,q_i,z_i)]$ 和 $\hat y_i^{(k)}\sim p_\theta(y\mid x_i,q_i,z_i;\tau,p)$，其中 $k=1,2,3$、温度 $\tau=0.7$、nucleus probability 为 90%、随机种子为 42。PathVQA、VQA-RAD 和 ProstateMM-CHIMERA 的最大新 token 分别为 24、32 和 48。清洗只移除 assistant 标签、答案前缀、多余空格和首个非空行之后的文本，不人工纠正内容。由于只有一次运行，结果不估计随机种子变化。

## 答案质量

greedy answer 经过小写化、标点移除和空格规范后，使用 BLEU-1、BLEU-2、ROUGE-L 与 METEOR 评分（Papineni 等，2002；Lin，2004；Banerjee 和 Lavie，2005）。BLEU 定义为 $\mathrm{BLEU}_N=\mathrm{BP}\exp(\sum_{n=1}^{N}w_n\log p_n)$；若 $L_i$ 为最长公共子序列，则 $P_{L,i}=L_i/|\hat y_i^{(0)}|$、$R_{L,i}=L_i/|y_i|$，ROUGE-L 是两者的加权调和平均。四个指标奖励不同性质，因此分别报告，不合并为单一准确率；计算仍使用标准归一化形式，展示统一使用百分数。

## 操作性失败定义

二元终点由 greedy answer 定义：$F_i=\mathbb I[\mathrm{ROUGE\mbox{-}L}_i<20\%\land\mathrm{METEOR}_i<10\%]$。联合条件和阈值在全部模型与数据集上固定。这是低参考一致性的可复现标签，不是专家确认的幻觉或临床错误，可能惩罚有效改写，也可能漏掉保留参考词汇的医学错误。标签来自 greedy answer、预测分数来自 sampled answers，因此同一组采样不会同时定义目标和预测器。

## 失败检测方法

保留三种非随机分数和一个随机基线，数值越大均表示不确定性越高；开发阶段计算的 answer disagreement 与语义条件重叠，因此不进入正式比较。

### AFD Frequency

对大小写、标点和 yes/no 形式规范化后，令 $c_i(a)$ 为答案 $a$ 在 $K=3$ 个样本中的频数，则 $u_i^{\mathrm{freq}}=1-\max_a c_i(a)/K$。该方法透明，但无法识别改写答案。

### Semantic AFD

使用 BAAI/bge-small-en-v1.5 对答案编码并进行 L2 归一化，余弦相似度至少为 80% 的样本相连，连通分量形成语义组 $\mathcal C_i$，分数为 $u_i^{\mathrm{sem}}=1-\max_{C\in\mathcal C_i}|C|/K$。它允许改写被视为一致，但依赖嵌入空间和阈值。

### Question Aligned Uncertainty

该分数结合平均问题答案对齐和采样答案间的一致性。使用归一化嵌入 $e(\cdot)$，先计算平均 question-answer cosine 与平均 answer-answer cosine，并缩放到 0%–100%，最终使用 $u_i^{\mathrm{qa}}=1-s_i^{\mathrm{qa}}s_i^{\mathrm{cons}}$。对齐或一致性较低都会提高不确定性。它继承 Carlini 等（2026）关于问题有效性的动机，但只是两个平均相似度的乘积，并不是 QA-SNNE 算法。
原始实现将该分数命名为 ``Question-aligned entropy''；由于它不是概率熵，论文统一使用 ``Question-aligned uncertainty''，以避免误解。

### 随机基线

使用种子 42 和样本索引生成 0%–100% 的确定性伪随机分数，表示不包含答案信息的排序；单次随机实现的 AUROC 不必恰好为 50%。

## 可靠性分析

AUROC 是主要检测指标，表示失败分数高于非失败分数的概率并对相等情况给一半权重；AUPRC 为次要指标，因为它受失败比例影响。先在每个模型和数据集内比较检测器，再把最佳 AUROC 模型排名与答案质量排名比较。选择性分析按不确定性从低到高排序，在覆盖率 $c$ 下接受前 $\lceil cN\rceil$ 个样本，计算接受集合的平均 ROUGE-L $Q_R(c)$ 和失败率 $\phi(c)$，并与相同覆盖率的随机排序比较。AUROC 衡量排序而非校准概率，本研究不选择临床操作阈值。

## 可复现性与伦理

实现固定模型标识、种子、数据划分、prompt、解码参数、规范化方式、指标库、失败阈值、嵌入模型、语义阈值和覆盖率，并按数据集与模型保存输出。本研究不收集新患者数据，ProstateMM-CHIMERA 的标识只用于保留分组信息。其主要伦理风险是过度陈述：针对词汇代理的高 AUROC 不能证明模型识别了临床有害答案，未来使用仍需要专家标签、患者层面统计、亚组评价、校准阈值和前瞻性工作流测试。
