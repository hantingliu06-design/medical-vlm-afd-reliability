# AFD Reliability Evaluation for Medical Vision-Language Models

本项目用于医学视觉问答（medical VQA）可靠性研究，比较 Qwen2.5-VL-3B-Instruct、MedGemma-4B-it 和 LLaVA-1.5-7B 在 PathVQA、VQA-RAD、ProstateMM/CHIMERA 上的 zero-shot 表现，并使用 AFD 及其他不确定性信号识别可能失败的答案。

项目流程：

```text
医学图像 + 问题 -> 三个 VLM 生成答案 -> 主答案质量评估
                                  -> K 个采样答案的不确定性
                                  -> Failure label -> AUROC/AUPRC/coverage
```

AFD 后处理读取已经保存的 JSON 推理结果，不会自动下载数据集或重新运行模型。

## 如何阅读

只想快速理解项目时，按以下顺序：

1. 本 README：研究目标、文件作用和执行顺序。
2. [PathVQA 最终结果](tmp/clean_final_pathvqa_afd_table.csv)。
3. [ProstateMM 最终结果](tmp/clean_final_prostatemm_afd_table.csv)。
4. [AFD 后处理脚本](AFD_Coverage_10_30_50_70_90_All_Datasets.py)。
5. [论文图表脚本](generate_dissertation_figures.py)。

想理解模型推理，阅读 [PathVQA 推理脚本](PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py)、[ProstateMM 推理脚本](ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py) 及其对应 Notebook。想理解论文，阅读 [latex_template/](latex_template/)、[references.bib](references.bib) 和 [OVERLEAF_IMPORT_GUIDE_CN.md](OVERLEAF_IMPORT_GUIDE_CN.md)。

## 文件说明

| 文件/目录 | 作用 |
| --- | --- |
| `PathVQA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py` | PathVQA test split 上三个 VLM 的 zero-shot 推理、采样、断点保存和答案评估。 |
| `ProstateMM_CHIMERA_Qwen_MedGemma_LLaVA_ZeroShot_AFD.py` | ProstateMM/CHIMERA 数据读取、三个 VLM 推理和可靠性评估，默认 test split。 |
| `AFD_Coverage_10_30_50_70_90_All_Datasets.py` | 从 JSON 输出计算 AFD frequency、Semantic AFD、answer disagreement、question-aligned entropy、random baseline、AUROC/AUPRC 和多种 coverage 指标。 |
| `*_AFD.ipynb` | 对应的 Google Colab/Jupyter 交互版本；`.py` 是建议维护的源码版本。 |
| `generate_dissertation_figures.py` | 从 `tmp/afd_summary_all_datasets_coverage_10_30_50_70_90.csv` 生成论文性能-拒答曲线和框架图。 |
| `generate_methodology_framework*.py` | 方法框架图的不同历史版本；通常只需保留最终版本。 |
| `tmp/*.csv` | 经过筛选的聚合结果表，不包含完整患者级输出。 |
| `results/README.md` | 结果字段、来源及复现注意事项。 |
| `data/README.md` | 数据集、模型访问、隐私和许可证说明。 |
| `docs/reproducibility.md` | 实验复现检查清单。 |
| `docs/upload_checklist.md` | GitHub 发布前检查清单。 |
| `latex_template/` | 论文 LaTeX 源文件、参考文献和图片。 |
| `references.bib` | 项目级 BibTeX 参考文献。 |
| `requirements.txt` | Python 依赖；PyTorch/CUDA 仍需按机器确认。 |
| `.gitignore` | 排除缓存、模型权重、原始数据、凭据和大文件。 |
| `CITATION.cff` | GitHub 引用信息模板，需要填入真实作者和仓库地址。 |

## 环境

推荐 Python 3.10/3.11、Linux/Google Colab 或 NVIDIA GPU 环境，并确保 CUDA、PyTorch、bitsandbytes 兼容。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

也可以不激活环境直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

模型使用 4-bit 量化时需要兼容的 GPU 依赖。不要把 token、数据集压缩包、患者记录、模型缓存放入 Git。

## 复现 PathVQA 或 ProstateMM

### 1. 准备数据和模型

从官方来源获取数据集，从 Hugging Face 获取模型访问权限，并将 token 放入 Colab Secrets 或环境变量。准备 Google Drive 或其他可写磁盘保存模型缓存和 JSON 输出。

### 2. 运行推理

先做小规模 smoke test，再跑完整 test split。打开对应脚本/Notebook，检查：

- `INSTALL_DEPENDENCIES`
- `RUN_ID`
- `DATASET_SPLIT` 或 `SPLIT_SELECTION`
- `MODEL_BATCH_SIZES`
- `K` 或 sampled-answer 数量
- Google Drive 输出目录和模型 ID

第一次安装依赖并重启运行时后，将 `INSTALL_DEPENDENCIES = False`，再从头执行。推理脚本会保存可断点恢复的 JSON 输出，请保留它们供后续 AFD 分析使用。

### 3. 运行 AFD 后处理

打开 `AFD_Coverage_10_30_50_70_90_All_Datasets.py`，检查 `MY_DRIVE_DIR`、`OUTPUT_DIRS` 和 `AFD_OUTPUT_DIR`。脚本默认选择每个数据集/模型组合中最大的完整运行，避免混合 smoke test 和最终实验。

它会计算 AFD frequency、Semantic AFD、answer disagreement、question-aligned entropy、random baseline、Failure AUROC、Failure AUPRC，以及 10%、30%、50%、70%、90% coverage 下的指标。只将经过审查的聚合 CSV 放入 `tmp/` 或 `results/`。

### 4. 生成论文图表

```powershell
python generate_dissertation_figures.py
```

脚本读取 `tmp/afd_summary_all_datasets_coverage_10_30_50_70_90.csv`，并把图片写入 `latex_template/Images/`。

## 当前复现边界

当前仓库可以查看和复现：已提交的聚合结果、PathVQA 推理、ProstateMM/CHIMERA 推理、AFD 后处理和论文图表生成。

仓库目前没有单独的 VQA-RAD 推理脚本。因此 VQA-RAD 汇总结果可以查看，但从原始数据完整重跑 VQA-RAD 需要补充对应的数据加载、模型推理和输出脚本。结果还会受到模型 revision、数据集版本、GPU/CUDA/PyTorch、batch size、量化、随机种子和原始 JSON 版本影响。正式实验请记录代码 commit、模型 revision、数据版本、硬件、依赖版本、seed、batch size、量化方式和 `K`。

## 数据、伦理和许可证

本项目不公开分发原始医学影像、患者级记录、模型权重或数据集压缩包。所有数据集和预训练模型必须遵守其原始许可证、数据使用协议和伦理要求。

发布前请：

1. 删除 Notebook 中的 token、私人路径和调试输出。
2. 确认结果表没有患者标识或可回溯信息。
3. 替换 [CITATION.cff](CITATION.cff) 中的作者、机构和仓库地址。
4. 根据代码、数据集和模型许可选择项目许可证。
5. 创建 Git tag，并在 release note 中记录实验版本。

## 引用

使用本项目代码或结果时，请引用 [CITATION.cff](CITATION.cff)，并同时引用 PathVQA、VQA-RAD、ProstateMM/CHIMERA 以及所使用模型的原始论文。

