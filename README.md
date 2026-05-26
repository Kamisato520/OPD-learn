# OPD Learn

中文 | [English](#english)

## 中文

OPD Learn 是一个静态 HTML 学习站，整理 On-Policy Distillation（OPD）的直觉、公式推导、核心论文路线，以及图像生成模型后训练阶段如何使用 OPD。

### 入口

部署到 GitHub Pages 后，访问站点根路径即可打开：

- `index.html`：总入口，串联所有本地 HTML 页面。
- `awesome-on-policy-distillation/opd-visual-guide.html`：OPD 快速可视化导览。
- `awesome-on-policy-distillation/opd-study-guide.html`：OPD 详细学习网页。
- `awesome-on-policy-distillation/opd-formula-derivation.html`：OPD 公式推导。
- `awesome-on-policy-distillation/opd-image-post-training/index.html`：图像生成 OPD 后训练研究站。

### 内容结构

- 基础理解：SFT、KD、OPD、RL 的状态分布差异。
- 公式推导：reverse KL、GKD、ExOPD、diffusion / flow OPD。
- 论文地图：MiniLLM、GKD、ExOPD、Revisiting OPD、Rethinking OPD 等。
- 图像生成：DiMO、pi-Flow、D-OPSD、Flow-OPD、DiffusionOPD、AnyFlow。
- 后训练配方：student rollout、teacher guidance、trajectory loss、诊断指标。

### GitHub Pages 部署

1. 打开仓库 **Settings**。
2. 进入 **Pages**。
3. Source 选择 `Deploy from a branch`。
4. Branch 选择 `main`，目录选择 `/ (root)`。
5. 保存后等待 GitHub Pages 完成发布。

### 说明

- 本仓库只保留 Markdown、HTML 和相关页面 assets（CSS、JS、图片）。
- 不包含本地 PDF 文件；HTML 中的论文 PDF 链接已改为 arXiv 或 HuggingFace 外部链接。
- 页面为静态文件，不需要构建步骤。
- 公式页面通过 MathJax CDN 渲染。
- `claude/DESIGN.md` 保留为页面设计参考。

---

## English

OPD Learn is a static HTML learning site for On-Policy Distillation (OPD). It collects intuition, formula derivations, key paper routes, and notes on applying OPD during image-generation model post-training.

### Entry Points

After deploying with GitHub Pages, open the site root:

- `index.html`: main hub linking all local HTML pages.
- `awesome-on-policy-distillation/opd-visual-guide.html`: compact OPD visual guide.
- `awesome-on-policy-distillation/opd-study-guide.html`: detailed OPD study guide.
- `awesome-on-policy-distillation/opd-formula-derivation.html`: OPD formula derivations.
- `awesome-on-policy-distillation/opd-image-post-training/index.html`: image-generation OPD post-training research hub.

### Content

- Foundations: state-distribution differences among SFT, KD, OPD, and RL.
- Formulas: reverse KL, GKD, ExOPD, diffusion / flow OPD.
- Paper map: MiniLLM, GKD, ExOPD, Revisiting OPD, Rethinking OPD, and more.
- Image generation: DiMO, pi-Flow, D-OPSD, Flow-OPD, DiffusionOPD, AnyFlow.
- Post-training recipe: student rollout, teacher guidance, trajectory loss, and diagnostics.

### GitHub Pages Deployment

1. Open repository **Settings**.
2. Go to **Pages**.
3. Set **Source** to `Deploy from a branch`.
4. Select branch `main` and folder `/ (root)`.
5. Save and wait for GitHub Pages to publish.

### Notes

- This repository keeps only Markdown, HTML, and related page assets: CSS, JS, and images.
- Local PDF files are not included. Paper PDF links in the HTML point to arXiv or HuggingFace instead.
- The site is static and requires no build step.
- Formula pages use the MathJax CDN.
- `claude/DESIGN.md` is included as the design reference.
