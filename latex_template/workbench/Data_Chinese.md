# 第三章：数据

本节说明三个数据集，它们分别进入同一个评价框架并独立计算结果，避免规模更大的 PathVQA 支配其他场景。

## 数据集选择

PathVQA 提供大规模组织病理基准，VQA-RAD 把图像领域改变为放射影像，ProstateMM-CHIMERA 则在前列腺评估任务中加入患者关联的临床背景。图像类型、问题形式、答案长度和背景信息的差异允许检验模型和检测器排名是否稳定，而不赋予任何数据集更高地位。这种比较研究数据集依赖性，但不能证明临床泛化。

| 数据集 | 图像场景 | 分析单位 | 测试记录数 |
|---|---|---|---:|
| PathVQA | 组织病理 | 图像问题对 | 6,719 |
| VQA-RAD | 放射影像 | 图像问题对 | 451 |
| ProstateMM-CHIMERA | 前列腺评估 | 患者关联任务记录 | 42 |

## PathVQA

PathVQA 包含病理图像以及关于器官、组织、形态和疾病概念的问题（He 等，2020）。实验使用完整官方测试划分的 6,719 个图像问题对，不使用任何测试样本进行适配或模型选择。每条记录提供图像、问题和参考答案，三个模型接收相同顺序的记录。正式论文中的示例图体现了其局部视觉焦点和短答案形式，也说明有效医学术语表面形式不同会给词汇评价带来困难。

## VQA-RAD

VQA-RAD 包含放射影像以及由临床人员生成的封闭式和开放式问题（Lau 等，2018）。官方测试划分包含 451 个图像问题对，全部以零样本方式生成一个 greedy answer 和三个 sampled answers。它的模态、术语和答案分布与 PathVQA 不同，形成另一个独立的答案质量与失败检测场景。较小样本量要求谨慎解释接近的差异，而且该数据集不能代表完整放射报告流程。
最终逐条文件已单独完成一致性检查。每个模型文件都包含相同的 451 个索引（0--450），没有空的 greedy answer，并且每条记录都有三个 sampled answers。文件使用 `qwen2_5_vl_3b`、`llava_1_5_7b` 和 `medgemma_4b_it` 三个模型标识。

## ProstateMM-CHIMERA

ProstateMM-CHIMERA 是本项目根据 CHIMERA 2025 Task 1 前列腺癌生化复发预测任务构建的评价包（CHIMERA 2025，2025），并不被描述为以该名称单独正式发表的数据集。记录组合了组织病理图像、定向问题和结构化临床背景。完整数据包含 95 名患者的 285 条任务记录，其中训练、验证和测试分别为 201、42 和 42 条，每名患者三条记录。本论文只使用来自 14 名患者的测试记录，因此结果是任务记录层面的估计，而不是相互独立的患者层面估计。正式论文中的示例图展示了更接近病理决策的答案形式。

## Prompt 与处理

每个数据集使用一个领域专用 system prompt，并在 Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B 和 MedGemma-4B-it 之间固定。prompt 规定可用证据并要求简洁答案；ProstateMM-CHIMERA 加入临床背景，但 target facts 和参考答案不会传入模型。

### PathVQA Prompt

> You are a professional pathologist specialized in histopathology. You are answering visual questions based on pathology slide images. Use the given image, the question, and appropriate pathology knowledge to answer. For yes/no questions, answer only 'yes' or 'no'. For other questions, answer with the shortest medically appropriate phrase or term. Do not repeat the question. Do not provide explanations or unrelated information.

### VQA-RAD Prompt

> You are a medical imaging specialist experienced in radiology. You are answering visual questions based on radiological images. Use the image, the question, and appropriate radiological knowledge to answer. For yes/no questions, answer only 'yes' or 'no'. For other questions, provide a concise radiological answer using a short phrase. Do not repeat the question. Do not add uncertain diagnosis, treatment advice, or unrelated details.

### ProstateMM-CHIMERA Prompt

> You are a specialist pathologist experienced in prostate cancer assessment. You are answering visual questions based on prostate histopathology images and the provided clinical context. Use only the image, the question, and the clinical context to answer. For biochemical recurrence prediction or other yes/no questions, answer only 'yes' or 'no'. For grading, extent assessment, or other open-ended questions, answer with the most concise clinically appropriate phrase or value. Do not repeat the question. Do not provide long explanations, reasoning steps, or unrelated information.

生成文本只移除常见 assistant 前缀、规范空格并保留首个非空行；大小写和标点只在评分时规范，预测答案不会被改写为参考答案形式。

## 记录构建与数据质量

PathVQA 和 VQA-RAD 的分析单位是图像问题对，ProstateMM-CHIMERA 的单位是患者关联任务记录。三个模型评价相同源记录，共产生 20,157 个 PathVQA、1,353 个 VQA-RAD 和 126 个 ProstateMM-CHIMERA greedy model-item 输出，每个输出配三个采样答案。源索引保证答案、标签和检测分数正确连接；评分前检查图像路径、必需字段、划分、缓存规模和重复记录，PathVQA 最终使用标记为 `test_full` 的记录，ProstateMM-CHIMERA 使用 `test`。完整的 VQA-RAD 逐条导出允许直接计算失败比例，而覆盖率行仍然只是选择性运行点，不能当作总体失败比例；小规模前列腺测试和患者内依赖仍是明确限制。
