# 第四章：结果

本章依次报告答案质量、操作性失败检测以及两者在三个数据集中的关系。所有有界数值用百分数显示，差异使用百分点；底层计算仍使用标准归一化形式。由于只有一次推理运行，且没有置信区间和显著性检验，结果均为描述性证据。

## 总体实验结果

总体结果显示两种排名的关系并不恒定。PathVQA 的答案排名取决于指标：Qwen2.5-VL-3B-Instruct 在 BLEU-1 和 BLEU-2 上领先，MedGemma-4B-it 在 ROUGE-L 和 METEOR 上领先，而最高检测 AUROC 属于 Qwen2.5-VL-3B-Instruct。VQA-RAD 的分离更加清晰，MedGemma-4B-it 领先三个答案指标，Qwen2.5-VL-3B-Instruct 则在失败检测中领先。ProstateMM-CHIMERA 出现一致关系，MedGemma-4B-it 同时领先三个答案指标和 AUROC。小规模、患者关联的前列腺结果需要特别谨慎解释。

| 数据集 | 答案质量领先者 | 最佳失败检测 | AUROC | 关系 |
|---|---|---|---:|---|
| PathVQA | Qwen2.5-VL-3B-Instruct 领先 BLEU；MedGemma-4B-it 领先 ROUGE-L/METEOR | Qwen2.5-VL-3B-Instruct；question-aligned | 81.20% | 指标依赖 |
| VQA-RAD | MedGemma-4B-it 领先三个指标 | Qwen2.5-VL-3B-Instruct；Semantic AFD | 69.43% | **分离** |
| ProstateMM-CHIMERA | MedGemma-4B-it 领先三个指标 | MedGemma-4B-it；question-aligned | 93.88% | **一致** |

## PathVQA 答案质量

PathVQA 上，Qwen2.5-VL-3B-Instruct 的 BLEU-1 为 25.29%、BLEU-2 为 1.38%，均为最高；MedGemma-4B-it 的 ROUGE-L 为 34.77%、METEOR 为 17.73%，均为最高；LLaVA-1.5-7B 四项均最低。与 Qwen2.5-VL-3B-Instruct 相比，MedGemma-4B-it 的 ROUGE-L 和 METEOR 分别高 3.03 和 1.63 个百分点，但 BLEU-1 和 BLEU-2 分别低 5.78 和 0.95 个百分点。因此，本数据集的答案强度必须结合具体指标解释，不能合并为一个通用准确率。

| 数据集 | 模型 | BLEU-1 | BLEU-2 | ROUGE-L | METEOR |
|---|---|---:|---:|---:|---:|
| PathVQA | Qwen2.5-VL-3B-Instruct | **25.29%** | **1.38%** | 31.74% | 16.10% |
|  | LLaVA-1.5-7B | 19.89% | 0.74% | 26.75% | 13.53% |
|  | MedGemma-4B-it | 19.51% | 0.43% | **34.77%** | **17.73%** |
| VQA-RAD | Qwen2.5-VL-3B-Instruct | 39.37% | **8.49%** | 48.16% | 25.94% |
|  | LLaVA-1.5-7B | 38.41% | 4.88% | 40.25% | 20.42% |
|  | MedGemma-4B-it | **40.18%** | 6.60% | **58.66%** | **31.05%** |
| ProstateMM-CHIMERA | Qwen2.5-VL-3B-Instruct | 25.54% | 8.77% | 19.56% | **22.48%** |
|  | LLaVA-1.5-7B | 19.00% | 2.18% | 15.27% | 13.90% |
|  | MedGemma-4B-it | **62.24%** | **28.57%** | **21.86%** | 16.77% |

## VQA-RAD 答案质量

VQA-RAD 上，MedGemma-4B-it 领先 BLEU-1、ROUGE-L 和 METEOR，Qwen2.5-VL-3B-Instruct 领先 BLEU-2，LLaVA-1.5-7B 四项均最低。MedGemma-4B-it 的 ROUGE-L 为 58.66%，分别比 Qwen2.5-VL-3B-Instruct 和 LLaVA-1.5-7B 高 10.50 和 18.41 个百分点，METEOR 的 31.05% 也为最高。该数据集内部可以把 MedGemma-4B-it 视为三个指标上的答案质量领先者，但其绝对分数高于 PathVQA 并不说明放射影像任务更容易，因为问题和参考答案分布不同。

## 操作性失败分布

最终逐条导出允许报告完整测试集上的操作性失败比例。该比例表示 greedy answer 同时低于两个固定参考匹配阈值的记录占比，不是临床错误率。MedGemma-4B-it 在三个数据集上的比例都最低。失败负担描述的是错误数量，失败检测描述的是排序能力，二者不能混为一谈；AUPRC 也会受到正类比例影响。完整测试集比例为：PathVQA 中 Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B 和 MedGemma-4B-it 分别为 66.88%、71.96% 和 62.94%；VQA-RAD 中分别为 47.23%、58.09% 和 36.81%；ProstateMM-CHIMERA 中分别为 54.76%、42.86% 和 33.33%。

## 失败检测结果

PathVQA 上，Qwen2.5-VL-3B-Instruct 的三个非随机检测器 AUROC 均为模型间最高，question-aligned uncertainty 达到 81.20%；LLaVA-1.5-7B 和 MedGemma-4B-it 的最佳 AUROC 均来自 AFD frequency，分别为 69.55% 和 68.63%。随机 AUROC 接近 50%。在 50% 覆盖率下，九个模型数据集最佳检测条件均比随机排序提高接受 ROUGE-L 并降低失败率，其中 Qwen2.5-VL-3B-Instruct 在 PathVQA 上分别改善 21.65 和 21.52 个百分点。折线图进一步显示，ProstateMM-CHIMERA 上 question-aligned uncertainty 对三个模型均有改善，而 AFD frequency 对 Qwen2.5-VL-3B-Instruct 和 LLaVA-1.5-7B 可能低于随机，说明检测器必须按数据集选择。

| 数据集 | 模型 | 最佳检测器 | AUROC | AUPRC |
|---|---|---|---:|---:|
| PathVQA | Qwen2.5-VL-3B-Instruct | question-aligned | **81.20%** | **90.71%** |
|  | LLaVA-1.5-7B | AFD frequency | 69.55% | 82.93% |
|  | MedGemma-4B-it | AFD frequency | 68.63% | 75.61% |
| VQA-RAD | Qwen2.5-VL-3B-Instruct | Semantic AFD | **69.43%** | 69.52% |
|  | LLaVA-1.5-7B | Semantic AFD | 63.67% | **73.53%** |
|  | MedGemma-4B-it | question-aligned | 66.03% | 61.47% |
| ProstateMM-CHIMERA | Qwen2.5-VL-3B-Instruct | question-aligned | 81.69% | 87.59% |
|  | LLaVA-1.5-7B | question-aligned | 87.27% | 86.12% |
|  | MedGemma-4B-it | question-aligned | **93.88%** | **89.59%** |

## 不同评价维度下的模型排名

PathVQA 的最佳 AUROC 顺序为 Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B、MedGemma-4B-it，与 BLEU 顺序一致，却与 MedGemma-4B-it 领先的 ROUGE-L 和 METEOR 冲突。最清楚的对比是 Qwen2.5-VL-3B-Instruct 与 MedGemma-4B-it：后者在两个更广义的答案对齐指标上领先，前者的最佳 AUROC 却高 12.57 个百分点。LLaVA-1.5-7B 的答案质量最低，但 PathVQA 最佳 AUROC 仍略高于 MedGemma-4B-it。平均答案质量由此不能决定失败排序能力。

## Overall Validation

VQA-RAD 再次支持分离：MedGemma-4B-it 领先三个答案指标，但 Qwen2.5-VL-3B-Instruct 的 AUROC 最高，为 69.43%；在 50% 覆盖率下，其 Semantic AFD 把接受 ROUGE-L 从 48.07% 提高到 60.55%，把失败率从 47.35% 降到 35.84%。ProstateMM-CHIMERA 则反转这一关系，MedGemma-4B-it 同时领先三个答案指标和 93.88% 的检测 AUROC；其 question-aligned detector 把 50% 覆盖率下的 ROUGE-L 从 27.71% 提高到 39.39%，失败率从 28.57% 降到 0.00%。后一个结果只基于 21 条接受记录，且最佳检测器也随场景改变，因此没有模型或方法在所有条件下占优。

## 代表性定量案例

三个经过验证的 50% 覆盖率案例都显示：与同样规模的随机子集相比，所选检测器提高接受 ROUGE-L 并降低失败率。PathVQA 中 Qwen2.5-VL-3B-Instruct 的 question-aligned 分数带来 21.65 和 21.52 个百分点改善；VQA-RAD 中其 Semantic AFD 带来 12.48 和 11.51 个百分点改善；ProstateMM-CHIMERA 中 MedGemma-4B-it 的 question-aligned 分数带来 11.68 个百分点 ROUGE-L 改善和 28.57 个百分点失败率下降。这些是汇总层面的操作点案例，不是单个临床病例；只有在完整逐条记录可验证后，才能加入图像、问题和答案案例。

| 数据集 | 模型与检测器 | 接受记录 | 接受 ROUGE-L：随机 → 检测器 | 失败率：随机 → 检测器 |
|---|---|---:|---:|---:|
| PathVQA | Qwen2.5-VL-3B-Instruct；question-aligned | 3,360 | 32.20% → 53.85% | 66.40% → 44.88% |
| VQA-RAD | Qwen2.5-VL-3B-Instruct；Semantic AFD | 226 | 48.07% → 60.55% | 47.35% → 35.84% |
| ProstateMM-CHIMERA | MedGemma-4B-it；question-aligned | 21 | 27.71% → 39.39% | 28.57% → 0.00% |

## 假设与目标支持

| 假设或目标 | 使用的证据 | 主要结果 | 解释 |
|---|---|---|---|
| H1：答案质量与失败意识可能分离 | 跨数据集答案质量和检测器排名 | VQA-RAD 与 PathVQA 的领先者不同，ProstateMM-CHIMERA 则一致 | 有条件支持 |
| H2：检测器表现依赖模型和数据集 | 三模型、四种保留方法的 AUROC/AUPRC | 最佳检测器随模型和数据集组合改变 | 描述性支持 |
| H3：有效检测器改善选择性运行 | 匹配覆盖率下的接受 ROUGE-L 和失败率 | 选择后的 50% 运行点优于随机排序 | 描述性支持 |
| 目标：分别比较答案效用和可靠性 | greedy answer 与 sampled answers 使用独立路径 | 不构造合并分数 | 已完成 |

该支持表是描述性的。研究只有一次推理运行、一个随机种子和一个由指标定义的 failure endpoint，因此不能视为统计学确证或临床验证。

## 结果总结

结果回答了三个层面的问题。第一，答案质量具有指标依赖性，尤其在 PathVQA 上。第二，即使 MedGemma-4B-it 在更广义的答案对齐指标上领先，Qwen2.5-VL-3B-Instruct 仍在 PathVQA 和 VQA-RAD 上获得最高失败检测 AUROC。第三，这种分离不是普遍规律，因为 MedGemma-4B-it 在较小的 ProstateMM-CHIMERA 测试中同时领先两个维度。Question-aligned uncertainty 同样具有条件性：它在 Qwen2.5-VL-3B-Instruct 的 PathVQA 条件和全部前列腺条件中最佳，却并非每个 PathVQA 或 VQA-RAD 模型的最佳方法。因此，更好的答案生成不必然意味着更好的失败意识，两种维度必须分别报告。
