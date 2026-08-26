# Overleaf 导入说明

## 推荐方式：新建项目并上传压缩包

1. 登录 Overleaf，回到项目列表页面。
2. 点击 **New Project**，再选择 **Upload Project**。
3. 上传 `MSc_Dissertation_Overleaf_Ready_2026-08-22_v2.zip`。
4. 项目打开后，进入左下角 **Settings**（或左上角 **File** 菜单中的设置）。
5. 在 **Compiler** 设置中确认：
   - **Main document**：`Main.tex`
   - **Compiler**：`pdfLaTeX`
6. 点击 **Recompile**。

压缩包的根目录直接包含 `Main.tex`，不要在解压后再把外层文件夹整体套入 Overleaf 项目。

## 导入到现有项目

为了保留旧版本，建议先在 Overleaf 中复制当前项目作为备份。然后解压本压缩包，将根目录中的 `.tex`、`.bib` 和 `.bst` 文件上传到现有项目根目录，并将 `Images` 文件夹中的文件上传到现有项目的 `Images` 文件夹。出现同名文件时，确认替换。最后将主文档设为 `Main.tex` 并重新编译。

## 文件结构

```text
Main.tex
_Packages.tex
Abstract.tex
CaseHypothesisAimsObjectives.tex
ImpactStatement.tex
Introduction.tex
Background.tex
Methodology.tex
Data.tex
Results.tex
Discussion.tex
Conclusions.tex
Appendix.tex
References.bib
elsarticle-num.bst
splncs04.bst
Images/
  uom_logo.pdf
  reliability_centred_vlm_framework.pdf
  reliability_centred_vlm_framework.png
  performance_rejection_qwen.pdf
  performance_rejection_llava.pdf
  performance_rejection_medgemma.pdf
```

论文的五个编号章节现在与评分标准对应：

1. `Introduction and Literature Review`
2. `Case, Hypothesis, Aims and Objectives`
3. `Design of Study or Method`
4. `Results`
5. `Discussion`（其中包含 `Conclusion` 小节）

`Presentation and Referencing` 是贯穿全文的评分项，由标题页、字体、页边距、行距、图表格式、文内引用和参考文献共同体现，因此没有额外增加一个空的内容章节。

## 常见问题

- 如果 Overleaf 找不到章节，检查对应 `.tex` 文件是否与 `Main.tex` 位于同一层。
- 如果图片缺失，检查 `Images` 文件夹名称的大小写和文件层级。
- 如果参考文献没有立即出现，先连续编译两次；Overleaf 通常会自动执行所需的 BibTeX 编译步骤。
- 如果编译的不是整篇论文，重新确认 **Main document** 是否为 `Main.tex`。
