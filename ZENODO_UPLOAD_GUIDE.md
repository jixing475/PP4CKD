# PP4CKD Zenodo 归档与上传操作指南 (Computer-Use / 自动化表单友好版)

> 本文档整理了将 `pp4ckd_models.tar.gz` 归档至 [Zenodo](https://zenodo.org/) 的完整元数据与操作清单。所有字段名与内容均按照 Zenodo 标准表单键值对排布，支持人工手动核对或通过 Computer-Use / 浏览器自动化脚本直接填写。

---

## 0. 准备物料与本地文件路径

* **上传目标文件**：`/Users/zero/Desktop/zeroverse/projects/proj_CKDdb/pp4ckd/dist/pp4ckd_models.tar.gz`
* **文件体积**：约 2.5 GB
* **内容物说明**：包含基于 ChEMBL 36 全量数据（155.3 万化合物、7,676 靶点）自主训练的 7 大分子指纹模型（ECFP4, Fused, ECFP6, AtomPair, Layered, RDKit, MHFP6）的完整 70 折交叉验证权重与 7 个全量生产模型权重（共 77 个 PyTorch `.pt` 文件）。
* **Zenodo 存款入口 (Deposit URL)**：[https://zenodo.org/deposit/new](https://zenodo.org/deposit/new)

---

## 1. Zenodo 核心表单填写清单 (键值对清晰对应)

### 1.1 基本类型 (Basic Information)
* **Resource type (资源类型)**: `Dataset` (数据集) 或 `Software` (软件)
* **Publication date (发布日期)**: `2026-10-05` (当天日期)
* **Title (标题)**:
  ```text
  PP4CKD: Pre-trained Deep Learning Models for Polypharmacology Prediction on ChEMBL 36
  ```

### 1.2 作者信息 (Creators / Authors)
* **Author 1**:
  * **Family name (姓)**: `Liu`
  * **Given name (名)**: `Jixing`
  * **Affiliation (机构)**: `Centre for Artificial Intelligence Driven Drug Discovery, Faculty of Applied Sciences, Macao Polytechnic University, Macao SAR, China`
  * **ORCID** (如有): 选填
* **Author 2 (通讯作者)**:
  * **Family name (姓)**: `Li`
  * **Given name (名)**: `Shu`
  * **Affiliation (机构)**: `Centre for Artificial Intelligence Driven Drug Discovery, Faculty of Applied Sciences, Macao Polytechnic University, Macao SAR, China`
* **Author 3**:
  * **Family name (姓)**: `Tong`
  * **Given name (名)**: `Henry H. Y.`
  * **Affiliation (机构)**: `Centre for Artificial Intelligence Driven Drug Discovery, Faculty of Applied Sciences, Macao Polytechnic University, Macao SAR, China`

### 1.3 描述与摘要 (Description)
*(支持直接复制粘贴至 Description 富文本框，或选择 Markdown 模式输入)*

```markdown
### Overview
This repository contains the complete pre-trained deep neural network weights (77 PyTorch model checkpoints, ~2.5 GB) for **PP4CKD**, a high-performance multi-target ligand-based target prediction and drug repositioning framework developed for chronic kidney disease (CKD) drug discovery.

### Training & Dataset Specifications
- **Data Source**: ChEMBL 36 SQLite release (standardized and filtered).
- **Dataset Scale**: 1,553,766 non-isomeric, desalted bioactive compounds, 7,676 biological targets (filtered for ≥5 active compounds across 95 activity types with activity ≤ 10 µM or inhibition ≥ 50%), and 3,173,052 bioactivity pairs.
- **Model Architecture**: Two-layer deep neural networks (Dense 1000, ReLU, Dropout 0.2 -> Dense 500, ReLU, Dropout 0.2 -> Output Sigmoid 7676), optimized with Adam and binary cross-entropy.

### File Manifest (`pp4ckd_models.tar.gz`)
The compressed archive unpacks into the `models/` directory containing:
1. **Full Production Models (7 files)**:
   - `fused_full_model.pt` (ECFP4 + MHFP6 8192-bit)
   - `ecfp4_full_model.pt` (Morgan r=2, 4096-bit)
   - `ecfp6_full_model.pt` (Morgan r=3, 4096-bit)
   - `mhfp6_full_model.pt` (MinHash FP, 4096-bit)
   - `atompair_full_model.pt` (Atom Pair, 4096-bit)
   - `layered_full_model.pt` (Layered FP, 4096-bit)
   - `rdkit_full_model.pt` (RDKit Daylight-like, 4096-bit)
2. **10-Fold Cross-Validation Models (70 files)**:
   - 10 checkpoint files for each of the 7 fingerprint representations (`*_fold_0.pt` to `*_fold_9.pt`).

### Benchmark Performance Highlights (ChEMBL 36 10-Fold CV)
- **Fused Model**: Overall Top-1 Recall: 53.62% ± 1.30%, Top-10 Recall: 85.10% ± 1.07% (Micro AUPR: 0.3481).
- **CKD-Specific Target Subset (623 Targets)**: Top-10 Recall reaches **87.54% ± 0.89%** (Micro AUPR: 0.3851).
- **ECFP4 Model**: Overall Top-10 Recall: 84.37% ± 1.30%; CKD Subset Top-10 Recall: **86.80% ± 1.14%**.

### Code Repository & Usage
The open-source code, data extraction pipelines, and quickstart inference scripts are publicly hosted on GitHub: [https://github.com/jixing475/pp4ckd](https://github.com/jixing475/pp4ckd).
```

### 1.4 许可协议 (License)
* **License**: `Creative Commons Attribution 4.0 International` (`CC-BY-4.0`)
  *(或选择 MIT License / Open Access)*

### 1.5 关键词与分类标签 (Keywords & Subjects)
* **Keywords**:
  ```text
  Polypharmacology, Target Prediction, Drug Repositioning, Deep Learning, ChEMBL 36, Chronic Kidney Disease, CKDdb, Molecular Fingerprints, PyTorch
  ```

### 1.6 关联标识与代码仓库 (Related Identifiers)
* **Relation type (关系)**: `isSupplementTo` (作为……的补充) 或 `isDerivedFrom`
* **Identifier (标识符)**: `https://github.com/jixing475/pp4ckd`
* **Resource type**: `Software`

---

## 2. 详细上传操作流程 (Step-by-Step)

### Step 1: 登录与创建条目
1. 打开浏览器访问 [https://zenodo.org/](https://zenodo.org/) 并登录账号。
2. 点击右上角或导航栏中的 **「Upload」** 按钮，或直接打开新建上传链接：[https://zenodo.org/deposit/new](https://zenodo.org/deposit/new)。

### Step 2: 上传模型打包文件
1. 在 **Files** 区域点击 **「Choose files」**，或者直接拖拽本地文件：
   `/Users/zero/Desktop/zeroverse/projects/proj_CKDdb/pp4ckd/dist/pp4ckd_models.tar.gz`
2. 点击 **「Start upload」**，等待 2.5 GB 文件上传完毕并显示绿色对勾（由于直连 Zenodo 服务器，通常需要 2~5 分钟）。

### Step 3: 填写元数据 (直接映射第 1 节内容)
1. **Basic info**: 依次填入 Title、Publication date、Resource type (`Dataset`)。
2. **Creators**: 依次录入 Jixing Liu、Shu Li、Henry H. Y. Tong 及单位。
3. **Description**: 完整粘贴第 1.3 节的 Markdown 内容。
4. **License**: 确保选定 `CC-BY-4.0`。
5. **Keywords**: 依次添加关键词。
6. **Related identifiers**: 填入 GitHub 仓库地址 `https://github.com/jixing475/pp4ckd`。

### Step 4: 保存与预分配 DOI (Reserve DOI)
1. 点击页面底部的 **「Save」**（此时生成草稿 Draft，尚未公开发布）。
2. 在页面顶部会看到 **「DOI」** 模块，点击 **「Reserve DOI」**。
3. **关键操作**：Zenodo 会立刻生成一个预留的 DOI（格式如：`10.5281/zenodo.12345678`）。
   - **立即复制该 DOI**，可直接填入手稿与 README，不需要等待最终提交即可锁定该编号。

### Step 5: 发布 (Publish)
1. 确认文件校验和与所有信息准确无误。
2. 点击绿色的 **「Publish」** 按钮，在二次确认弹窗中确认发布。
3. 状态变更为 Published，该 DOI 即刻永久生效！

---

## 3. 发布后的联动回填清单 (仅需 2 处)

发布并取得 DOI（假设为 `10.5281/zenodo.XXXXXXX`）后，仅需更新以下两处引用：

1. **`pp4ckd/README.md`**：
   - 将 `https://doi.org/10.5281/zenodo.xxxxxxx` 替换为真实生成的 DOI 链接；
   - 更新顶部的 Zenodo Badge 徽标。
2. **`manuscript/draft/FULL_MANUSCRIPT.md`**：
   - 将 Data Availability Statement 中的 `[TODO-D1: Zenodo DOI for CSV release snapshot]` 关联处或 PP4CKD 权重获取处填入该 DOI。
