# 论文图表规范工作单

## Handbook 与曼大官方 presentation policy 的硬性要求

- 论文主文字号至少为 12 pt，并使用 1.5 倍或双倍行距。
- Figures and images 必须具有足够的尺寸和清晰度。
- 论文含有图表时，应在目录后分别提供 List of Figures 和 List of Tables，并为每一项列出页码。
- 图、表及其他材料都必须纳入论文连续的 Arabic page-number sequence。
- 图表标题中的文字计入 dissertation word count，但图表内部内容本身不计入。
- 使用第三方图片、图或表时，必须正确注明来源，并确认不存在版权限制或已经取得使用许可。
- Handbook 没有规定固定的图表配色、表格线型或 caption 上下位置；这些细节由论文模板和一致的学术排版规则控制。

来源：本项目提供的 *Dissertation Handbook: MSc Health Data Science*，第 24–25 页；曼大 [Guidance for the Presentation of Taught Dissertations](http://documents.manchester.ac.uk/display.aspx?DocID=2863)，第 3–4 页。

## 本论文采用的统一图表规则

- 正式插图优先使用 vector PDF；工作台同时保留 300 dpi PNG 预览。
- 折线图使用共同的坐标尺度，并同时使用颜色、线型和 marker 区分方法，避免只依赖颜色。
- 图 caption 放在图下方，表 caption 放在表上方；caption 应说明比较对象和读图方向，但避免重复正文。
- 所有图表必须先在正文中被引用，不能作为没有解释的独立装饰。
- 表格使用 `booktabs` 风格，不使用竖线，不使用依赖彩色打印的底色。
- 数值表按预先确定的比较组进行强调：**粗体表示最佳值**，下划线表示第二名；caption 必须说明比较范围。
- 不跨不同数据集比较绝对 answer-quality 数值的难度，只比较同一数据集内部的模型排名。
- ProstateMM-CHIMERA 始终标明其结果来自 42 条 task records 和 14 位患者，不能把记录层结果写成患者层临床性能。

## 当前正式图表

- Chapter 3 的 Data 部分：一张 dataset comparison table，以及 PathVQA、VQA-RAD 和 ProstateMM-CHIMERA 各一张代表性样本图。
- Methodology chapter：一张 reliability-centred evaluation framework；正式版使用 vector PDF，工作台使用 300 dpi PNG。
- Results chapter：七张比较表，依次覆盖总体结论、答案质量、失败负担、PathVQA detector、50% coverage 增益、跨数据集最佳 detector，以及三个数据集的代表性定量案例。
- Results chapter：三张 performance-rejection line figures，分别对应 Qwen2.5-VL-3B-Instruct、LLaVA-1.5-7B 和 MedGemma-4B-it；每张图包含 PathVQA 与 ProstateMM-CHIMERA 两个 panel。
