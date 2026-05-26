# OPD Start Here 学习文档

生成日期：2026-05-19  
范围：严格按照 `README.md` 的 `Start Here` 路径整理。论文 PDF 已保存在 `papers/start-here/`；Thinking Machines 博客作为辅助阅读资料，不转存为 PDF。

## 0. 一句话定义

On-Policy Distillation（OPD）是在学生模型自己的采样轨迹上训练学生，同时让教师模型在学生实际访问到的前缀状态上提供稠密 token 级指导。它把蒸馏数据分布从“教师/数据集固定轨迹”切换到“学生当前策略轨迹”，因此主要解决 SFT 与离线 KD 在生成时常见的 exposure bias 和 train-inference distribution gap。

OPD 最适合下面几类场景：

| 场景 | 为什么适合 OPD | 代表资料 |
|---|---|---|
| 大模型压缩、强到弱迁移 | 小模型会走到教师数据之外的前缀；OPD 让教师在这些前缀上纠偏 | MiniLLM、GKD、Qwen3 |
| RL 后能力整合 | 多个领域专家或多阶段 checkpoint 难以直接合并；OPD 可把专家策略蒸馏到统一学生 | DeepSeek-V4、MiMo-V2-Flash、GLM-5 |
| 稳定替代稀疏奖励 RL | 教师 token 级 logit/log-prob 信号比 outcome reward 更稠密 | ExOPD、MiMo-V2-Flash |
| 无外部教师但有额外上下文 | 同一个模型在 privileged context、demonstration、experience 下可作为教师 | OPSD、SDFT、OPCD、OEL |
| API-only 教师 | 无法拿 logits 时，可用判别器或 reward 模型近似教师偏好 | Black-Box OPD |

OPD 与 SFT、KD、RL 的关系：

| 方法 | 数据来自谁 | 监督信号 | 典型目标 | OPD 的差异 |
|---|---|---|---|---|
| SFT | 人类/教师固定答案 | gold token NLL | `-\log \pi_\theta(y^*|x)` | 不看学生自己会犯什么错 |
| 离线 KD / SeqKD | 教师采样或数据集轨迹 | 教师 token 或序列 | 在固定轨迹上拟合教师 | 训练状态仍是 off-policy |
| OPD | 学生当前策略 rollout | 教师在学生前缀上的 dense guidance | 学生分布贴近教师分布 | 训练状态与推理状态一致 |
| RL / RLVR | 学生 rollout | reward，常为 outcome-level | 最大化奖励并约束 KL | OPD 可看作教师定义的 dense reward / KL-constrained RL |

## 1. 统一数学视角

### 1.1 Student rollout 与 teacher guidance

给定 prompt `x`，学生模型 `\pi_\theta` 先生成自己的轨迹：

$$
y=(y_1,\dots,y_T)\sim \pi_\theta(\cdot|x)
$$

在每个学生访问到的前缀 `c_t=(x,y_{<t})` 上，教师 `\pi_T` 提供 token 分布或 log-prob：

$$
\pi_T(\cdot|c_t),\quad \pi_\theta(\cdot|c_t)
$$

最常见的白盒 OPD 目标是对学生 rollout 上的 token 分布做 reverse KL：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{x,\,y\sim\pi_\theta(\cdot|x)}
\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}\left(
\pi_\theta(\cdot|c_t)\,\|\,\pi_T(\cdot|c_t)
\right)
\right].
$$

这个式子里的关键不是 KL 本身，而是期望分布 `y \sim \pi_\theta`：教师不是在自己的答案上教学生，而是在学生真实会到达的状态上教学生。

### 1.2 Reverse KL 与 forward KL

Reverse KL：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_T)
=
\sum_v \pi_\theta(v|c)\log\frac{\pi_\theta(v|c)}{\pi_T(v|c)}
$$

它惩罚学生把概率放到教师低概率 token 上，通常更 mode-seeking，适合压制错误模式、提高确定性和执行一致性。

Forward KL：

$$
D_{\mathrm{KL}}(\pi_T\|\pi_\theta)
=
\sum_v \pi_T(v|c)\log\frac{\pi_T(v|c)}{\pi_\theta(v|c)}
$$

它要求学生覆盖教师认为可能的 token，通常更 mode-covering，适合保留多样性。Entropy-Aware OPD 的核心就是在教师高熵状态下补 forward KL，避免 reverse KL 把不确定位置过早坍缩成单一 token。

### 1.3 Sampled-token estimator

若只拿学生实际采到的 token `y_t`，可把 reverse-KL 风格指导写成 token advantage：

$$
A_t
=
\log \pi_T(y_t|c_t)
-
\log \pi_\theta(y_t|c_t).
$$

对应的 policy-gradient 式损失常写成：

$$
\mathcal L_{\mathrm{sample}}
=
-
\mathbb E_{y\sim\pi_\theta}
\left[
\sum_t
\operatorname{sg}(A_t)\log \pi_\theta(y_t|c_t)
\right].
$$

其中 `sg` 表示 stop-gradient。直觉是：若教师比学生更支持学生采到的 token，则增加该 token；若教师不支持，则降低该 token。MiniLLM 与 MiMo-V2-Flash 都使用了类似思想，但后续论文指出 sampled-token estimator 方差和稳定性问题较明显。

### 1.4 GKD 混合策略

GKD 把 on-policy 与 off-policy 写成一个混合目标。设 `D` 是 token-level divergence，`λ` 是学生生成数据比例：

$$
\mathcal L_{\mathrm{GKD}}
=
(1-\lambda)
\mathbb E_{(x,y)\sim \mathcal D}
D(\pi_T,\pi_\theta;y,x)
+
\lambda
\mathbb E_{x,\,y\sim\pi_\theta(\cdot|x)}
D(\pi_T,\pi_\theta;y,x).
$$

当 `λ=0` 时接近传统离线 KD；当 `λ=1` 时是纯 OPD。GKD 的工程优势是可直接在 student rollout 上计算 token 分布差异，不必对采样过程反向传播，因此比早期 policy-gradient 式 OPD 更容易稳定实现。

### 1.5 Dense reward / KL-constrained RL

ExOPD 把 OPD 放进 KL-constrained RL 框架。标准形式可写成：

$$
\max_{\pi_\theta}
\mathbb E_{y\sim\pi_\theta}
\left[
r(x,y)
-
\beta D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))
\right].
$$

若令教师隐含 reward 为：

$$
r_T(x,y)=\log\frac{\pi_*(y|x)}{\pi_{\mathrm{ref}}(y|x)},
$$

OPD 就等价于使用教师定义的 dense reward。ExOPD 再引入尺度 `λ`：

$$
\max_{\pi_\theta}
\mathbb E_{y\sim\pi_\theta}
\left[
\lambda\log\frac{\pi_*(y|x)}{\pi_{\mathrm{ref}}(y|x)}
-
\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
\right].
$$

最优解满足：

$$
\log \pi_\theta(y|x)
=
\lambda \log \pi_*(y|x)
+
(1-\lambda)\log \pi_{\mathrm{ref}}(y|x).
$$

`0<λ<1` 是插值，`λ=1` 近似普通 OPD，`λ>1` 是 reward extrapolation，用来尝试超越教师。

### 1.6 Black-box reward 与 self/context distillation

当没有教师 logits 时，Black-Box OPD 用判别器 `D_\phi(x,y)` 学习区分教师回答和学生回答，再把判别器分数作为学生 on-policy reward：

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
D_\phi(x,y).
$$

当没有外部教师时，可让同一个模型在更强输入条件下充当教师。例如 OPSD 让教师看到参考推理 `y^*`，SDFT 让教师看到 demonstration `c`，OPCD 让教师看到额外 context `c`，而学生只看原始 prompt `x`。统一写法是：

$$
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
\sum_t
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x,y_{<t})
\|
\pi_T(\cdot|x,z,y_{<t})
\right),
$$

其中 `z` 是 privileged reasoning、demonstration、retrieved context 或 deployment experience。

## 2. README Start Here 的 8 步学习路径

### 第 1 步：Survey

先读 OPD Survey，目标是建立分类坐标：teacher access、反馈信号、loss scope、on-policy 采样方式、实验风险。读完后应能判断一个新方法属于白盒 logits 蒸馏、黑盒 reward、自蒸馏、上下文蒸馏，还是工业多教师整合。

阅读资料：

- [01-opd-survey-2604.00626.pdf](papers/start-here/01-opd-survey-2604.00626.pdf)

### 第 2 步：Foundations

读 MiniLLM、GKD、ExOPD。MiniLLM 解释为什么 generative LM 蒸馏更适合 reverse KL；GKD 给出更通用、更工程化的 on/off-policy 混合目标；ExOPD 把 OPD 解释为 dense KL-constrained RL，并引入 reward extrapolation。

阅读资料：

- [02-minillm-2306.08543.pdf](papers/start-here/02-minillm-2306.08543.pdf)
- [03-gkd-2306.13649.pdf](papers/start-here/03-gkd-2306.13649.pdf)
- [04-exopd-2602.12125.pdf](papers/start-here/04-exopd-2602.12125.pdf)

### 第 3 步：Practical intuition

读 Thinking Machines 的博客，重点理解为什么 OPD 不是“把 KD 换个名字”，而是把监督放到学生实际会访问的状态上。博客适合建立工程直觉：什么时候 OPD 值得上、什么时候普通 SFT/KD 已经足够、什么时候应当引入多教师或 reward。

阅读资料：

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

### 第 4 步：When OPD works and when it breaks

读 Revisiting OPD、Entropy-Aware OPD、Rethinking OPD。核心问题是 OPD 的成功条件与失败模式：sampled-token estimator 不稳定、reverse KL 导致高熵 token 多样性坍缩、教师在学生偏离前缀上的指导可能失真、强弱模型思维模式不兼容。

阅读资料：

- [05-revisiting-opd-2603.25562.pdf](papers/start-here/05-revisiting-opd-2603.25562.pdf)
- [06-entropy-aware-opd-2603.07079.pdf](papers/start-here/06-entropy-aware-opd-2603.07079.pdf)
- [07-rethinking-opd-2604.13016.pdf](papers/start-here/07-rethinking-opd-2604.13016.pdf)

### 第 5 步：No teacher logits

读 Black-Box OPD。问题变成：教师只有 API 生成能力，不能给 token logits，该如何保持 on-policy 学习？答案是用在线判别器把教师行为转成 reward，再对学生做 on-policy RL。

阅读资料：

- [08-black-box-opd-2511.10643.pdf](papers/start-here/08-black-box-opd-2511.10643.pdf)

### 第 6 步：No teacher at all

读 OPSD 与 SDFT。它们共同展示了“同一个模型在不同输入条件下自蒸馏”的范式：教师不一定是另一个更大模型，也可以是同模型加 privileged context、答案、示例或 EMA 权重。

阅读资料：

- [09-opsd-2601.18734.pdf](papers/start-here/09-opsd-2601.18734.pdf)
- [10-sdft-2601.19897.pdf](papers/start-here/10-sdft-2601.19897.pdf)

### 第 7 步：Context and experience

读 OPCD 与 OEL。OPCD 是把额外 context 蒸进权重；OEL 把这个思路扩展到部署循环，用用户端轨迹抽取经验，再用 OPCD 让模型在无上下文时也表现得像有经验。

阅读资料：

- [11-opcd-2602.12275.pdf](papers/start-here/11-opcd-2602.12275.pdf)
- [12-oel-2603.16856.pdf](papers/start-here/12-oel-2603.16856.pdf)

### 第 8 步：Industrial recipes

最后读 Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5。这里不再只看单一 loss，而要看完整 post-training pipeline：SFT、RL、专家模型、多阶段 checkpoint、teacher scheduling、logit 计算成本、能力遗忘与恢复。

阅读资料：

- [13-qwen3-2505.09388.pdf](papers/start-here/13-qwen3-2505.09388.pdf)
- [14-deepseek-v4.pdf](papers/start-here/14-deepseek-v4.pdf)
- [15-mimo-v2-flash-2601.02780.pdf](papers/start-here/15-mimo-v2-flash-2601.02780.pdf)
- [16-glm-5-2602.15763.pdf](papers/start-here/16-glm-5-2602.15763.pdf)

## 3. 逐篇论文卡片

### 01. OPD Survey

本地 PDF：[01-opd-survey-2604.00626.pdf](papers/start-here/01-opd-survey-2604.00626.pdf)  
原始链接：[https://arxiv.org/abs/2604.00626](https://arxiv.org/abs/2604.00626)

研究问题：OPD 方法已经从 MiniLLM/GKD 扩展到黑盒、自蒸馏、上下文蒸馏和工业多教师配方，需要一个统一分类框架来说明“什么算 OPD、差异在哪里、如何选型”。

核心方法：Survey 不是提出新 loss，而是建立 OPD 的坐标系。最关键的三条轴是 teacher access、feedback granularity、training distribution。teacher access 区分白盒 logits、黑盒 API、自教师、上下文教师；feedback granularity 区分 full-vocab KL、top-k KL、sampled-token log-prob、sequence reward；training distribution 区分 teacher/data rollout、student rollout、混合 rollout。

关键公式可用通用 OPD 目标概括：

$$
\min_\theta
\mathbb E_{x,\,y\sim q_\theta(\cdot|x)}
\left[
\sum_t
\mathcal D_t
\left(
\pi_T(\cdot|x,y_{<t}),
\pi_\theta(\cdot|x,y_{<t})
\right)
\right],
$$

其中 `q_\theta` 通常是学生当前策略或带温度/截断的学生采样策略，`\mathcal D_t` 可以是 KL、JSD、sampled-token advantage 或 learned reward。

训练流程：先用学生采样构造 on-policy 前缀，再让教师在这些前缀上计算指导信号，最后用 KL、reward 或 policy-gradient 更新学生。

优点：给 OPD 方法选型提供统一语言，避免把所有“蒸馏 + RL”混为一谈。

局限：Survey 本身不提供单一可复现实验配方；具体实现仍依赖教师访问权限、推理成本和任务类型。

与其他方法的关系：它是本学习路径的索引。后续 15 篇可以视为 Survey 坐标系中的具体点。

### 02. MiniLLM: On-Policy Distillation of Large Language Models

本地 PDF：[02-minillm-2306.08543.pdf](papers/start-here/02-minillm-2306.08543.pdf)  
原始链接：[https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)

研究问题：传统 KD 常用 forward KL 或在教师/数据轨迹上训练，但生成式 LLM 在推理时会进入学生自己的分布。若学生在错误前缀上无人纠偏，会出现暴露偏差和长文本退化。

核心方法：MiniLLM 把学生视为策略，让学生 on-policy 生成，再用教师 log-prob 形成 reverse-KL 风格的 policy-gradient 更新。它强调 reverse KL 更适合 generative LM，因为它会惩罚学生给教师低概率区域分配概率。

关键公式：

$$
\theta^*
=
\arg\min_\theta
D_{\mathrm{KL}}\left(q_\theta(y|x)\|p_T(y|x)\right)
=
\arg\min_\theta
\mathbb E_{y\sim q_\theta}
\left[
\log\frac{q_\theta(y|x)}{p_T(y|x)}
\right].
$$

对 sampled token，可得到近似 advantage：

$$
A_t
=
\log p_T(y_t|x,y_{<t})
-
\log q_\theta(y_t|x,y_{<t}).
$$

训练流程：学生生成响应；教师计算学生 token 的 log-prob；用 policy-gradient 更新学生；同时使用稳定化机制，包括单步分解、教师-学生混合采样、长度归一化、reward clipping 和语言模型辅助损失。

优点：第一次清晰命名并系统化 LLM OPD；解释了为什么 reverse KL 比 forward KL 更适合压缩生成模型；对长回答和指令跟随更稳。

局限：policy-gradient estimator 方差较大；只看 sampled token 会浪费 full-vocab logit 信息；需要教师白盒 log-prob。

与其他 OPD 方法的关系：GKD 将它推广为更直接的 divergence 最小化；Revisiting OPD 分析其 sampled-token 估计不稳定；MiMo-V2-Flash 在工业配方中沿用了 token advantage 思想并加入重要性截断。

### 03. GKD: On-Policy Distillation of Language Models

本地 PDF：[03-gkd-2306.13649.pdf](papers/start-here/03-gkd-2306.13649.pdf)  
原始链接：[https://arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649)

研究问题：如何用一个统一、工程上稳定的目标覆盖传统 KD、on-policy KD，以及两者混合？

核心方法：GKD 在学生生成的样本上计算教师和学生的 token-level divergence，但不对采样过程反向传播。它引入 `λ` 控制学生生成数据比例，并允许使用 forward KL、reverse KL 或 generalized JSD。

关键公式：

$$
D(p_T\|p_S^\theta)(y|x)
=
\frac{1}{|y|}
\sum_{n=1}^{|y|}
D\left(
p_T(\cdot|y_{<n},x)
\|
p_S^\theta(\cdot|y_{<n},x)
\right).
$$

纯 on-policy 目标：

$$
\mathcal L_{\mathrm{OD}}(\theta)
=
\mathbb E_{x,\,y\sim p_S(\cdot|x)}
\left[
D(p_T\|p_S^\theta)(y|x)
\right].
$$

混合目标：

$$
\mathcal L_{\mathrm{GKD}}
=
(1-\lambda)
\mathbb E_{(x,y)\sim(\mathcal X,\mathcal Y)}
D(p_T\|p_S^\theta)(y|x)
+
\lambda
\mathbb E_{x,\,y\sim p_S(\cdot|x)}
D(p_T\|p_S^\theta)(y|x).
$$

JSD 版本：

$$
D_{\mathrm{JSD}(\beta)}(P\|Q)
=
\beta D_{\mathrm{KL}}(P\|\beta P+(1-\beta)Q)
+
(1-\beta)D_{\mathrm{KL}}(Q\|\beta P+(1-\beta)Q).
$$

训练流程：从数据 prompt 采样；按比例混入 gold/teacher trajectory 与 student rollout；在对应前缀上算 divergence；更新学生。

优点：实现简单，稳定性好；可平滑过渡 off-policy 到 on-policy；可与 RL 目标组合。

局限：仍需要教师 logits；full-vocab 计算成本高；若教师在学生偏离前缀上不可靠，GKD 本身不解决该问题。

与其他 OPD 方法的关系：GKD 是后续白盒 OPD 的标准基线。OPSD、OPCD、SDFT 都可看作把 `p_T` 换成 privileged-context teacher 的 GKD 型目标。

### 04. ExOPD: Learning beyond Teacher via Generalized OPD

本地 PDF：[04-exopd-2602.12125.pdf](papers/start-here/04-exopd-2602.12125.pdf)  
原始链接：[https://arxiv.org/abs/2602.12125](https://arxiv.org/abs/2602.12125)

研究问题：普通 OPD 通常被理解为“学生贴近教师”。如果教师由 RL 得到，学生是否只能模仿教师，还是可以借助教师隐含 reward 超越教师？

核心方法：ExOPD 证明 OPD 是 dense KL-constrained RL 的一个特例。教师相对于 reference model 的 log-ratio 可解释为 reward；通过缩放该 reward，学生可以在策略空间中对教师行为做 extrapolation。

关键公式：

$$
r_T(x,y)
=
\log\frac{\pi_*(y|x)}{\pi_{\mathrm{ref}}(y|x)}.
$$

广义目标：

$$
\mathcal J_{\mathrm{G\text{-}OPD}}
=
\max_{\pi_\theta}
\mathbb E_{y\sim\pi_\theta}
\left[
\lambda r_T(x,y)
-
D_{\mathrm{KL}}
(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))
\right].
$$

闭式最优方向：

$$
\log\pi_\theta(y|x)
=
\lambda\log\pi_*(y|x)
+
(1-\lambda)\log\pi_{\mathrm{ref}}(y|x).
$$

训练流程：准备 expert teacher 与 reference；学生 on-policy 采样；计算 teacher-reference log-ratio reward；用 KL-constrained 目标训练学生；调节 `λ`，其中 `λ>1` 进入 ExOPD 区间。

优点：把 OPD 与 RL 的关系讲清楚；提供“超越教师”的理论路径；适合 multi-teacher 或 strong-to-weak 后处理。

局限：需要 reference model，且 reward correction 对 teacher/reference 的质量敏感；`λ>1` 可能放大教师偏差；工程上仍要控制 KL 与采样稳定性。

与其他 OPD 方法的关系：MiniLLM/GKD 讲“如何贴近教师”，ExOPD 讲“教师信号如何作为 reward 被缩放”。MiMo-V2-Flash 的 teacher KL reward + outcome reward 也接近这个视角。

### 05. Revisiting On-Policy Distillation

本地 PDF：[05-revisiting-opd-2603.25562.pdf](papers/start-here/05-revisiting-opd-2603.25562.pdf)  
原始链接：[https://arxiv.org/abs/2603.25562](https://arxiv.org/abs/2603.25562)

研究问题：为什么朴素 sampled-token OPD 在多任务或长序列上容易不稳定？token-level estimator 到底偏在哪里？

核心方法：论文分析 token-level 与 sequence-level reverse KL 的偏差和方差，并提出 local support matching（LSM）。LSM 不只看采样 token，而是在教师 top-K 支持集上比较教师和学生分布，降低特殊 token、低概率噪声和一 token 监督失衡带来的不稳定。

关键公式。完整局部 reverse KL：

$$
\mathcal L_{\mathrm{full}}(c_t)
=
\sum_v
\pi_\theta(v|c_t)
\log
\frac{\pi_\theta(v|c_t)}{q(v|c_t)}.
$$

教师支持集：

$$
S(c_t)=\mathrm{TopK}_{q(\cdot|c_t)}.
$$

在支持集上重归一化后优化截断 reverse KL：

$$
\mathcal L_{\mathrm{LSM}}(c_t)
=
D_{\mathrm{KL}}
\left(
\tilde \pi_\theta^{S}(\cdot|c_t)
\|
\tilde q^{S}(\cdot|c_t)
\right).
$$

训练流程：学生用 top-p 等策略 rollout；教师在学生前缀上给 logits；取教师 top-K 支持集；对教师与学生在支持集内的分布重归一化；计算局部 reverse KL；屏蔽特殊 token 并控制长度。

优点：比 sampled-token OPD 更稳定；利用教师分布结构而非单个 token；能缓解 tokenizer 和特殊 token mismatch。

局限：需要 teacher logits 和 top-K/full-vocab 访问；top-K 大小是重要超参；若教师在偏离前缀上整体不可靠，支持集匹配仍会学到错误指导。

与其他 OPD 方法的关系：它是 MiniLLM/GKD 的稳定性修正；与 Entropy-Aware OPD 一样都在修补 reverse KL 的实际训练问题。

### 06. Entropy-Aware OPD

本地 PDF：[06-entropy-aware-opd-2603.07079.pdf](papers/start-here/06-entropy-aware-opd-2603.07079.pdf)  
原始链接：[https://arxiv.org/abs/2603.07079](https://arxiv.org/abs/2603.07079)

研究问题：reverse KL 的 mode-seeking 特性会不会牺牲教师在不确定位置的多样性？在数学推理、采样多解等场景中，这种坍缩会降低 pass@k。

核心方法：用教师 entropy 判断当前 token 是否高不确定。低熵位置继续用 OPD/reverse KL；高熵位置额外加入 forward KL，让学生覆盖教师的可选 token。

关键公式。教师 entropy：

$$
H_t^{\mathrm{te}}
=
-
\sum_x
\pi_{\mathrm{te}}(x|c_t)
\log \pi_{\mathrm{te}}(x|c_t).
$$

Entropy-aware 目标：

$$
\mathcal L_{\mathrm{EOPD}}(\theta;c_t)
=
\mathcal L_{\mathrm{OPD}}(\theta;c_t)
+
\mathbf 1[H_t^{\mathrm{te}}>\tau]
\mathcal L_{\mathrm{FKL}}(\theta;c_t).
$$

Forward KL 项：

$$
\mathcal L_{\mathrm{FKL}}
=
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{te}}(\cdot|c_t)
\|
\pi_\theta(\cdot|c_t)
\right).
$$

训练流程：学生 rollout；教师计算 logits 与 entropy；若 entropy 超过阈值 `τ`，在 top-k teacher tokens 上近似 forward KL；否则主要使用 reverse-KL OPD。

优点：保留教师高熵位置的多样性；改善 pass@k 与探索；改动局部，容易叠加到现有 GKD/OPD 框架。

局限：需要教师 logits 和 entropy；阈值 `τ` 与 top-k 需要调参；如果高熵来自教师混乱而不是合理多样性，forward KL 会传播噪声。

与其他 OPD 方法的关系：Revisiting OPD 修补估计器局部支持，Entropy-Aware OPD 修补 reverse KL 的多样性问题；两者都属于白盒稳定化技术。

### 07. Rethinking On-Policy Distillation

本地 PDF：[07-rethinking-opd-2604.13016.pdf](papers/start-here/07-rethinking-opd-2604.13016.pdf)  
原始链接：[https://arxiv.org/abs/2604.13016](https://arxiv.org/abs/2604.13016)

研究问题：为什么同样是更强教师，有时 OPD 成功，有时失败？教师 benchmark 分数高是否足够？

核心方法：论文提出 OPD 成功的两个条件。第一，学生与教师需要兼容的 thinking patterns，即 top-k token 分布有足够重叠。第二，教师必须提供学生已有能力之外的新知识或新策略，而不只是分数更高。

关键诊断公式。top-k overlap：

$$
M_{\mathrm{overlap}}
=
\mathbb E_t
\left[
\frac{|S_t(\pi_\theta)\cap S_t(\pi_T)|}{k}
\right].
$$

Entropy gap：

$$
\Delta H_t
=
\left|
H(\pi_T(\cdot|c_t))
-
H(\pi_\theta(\cdot|c_t))
\right|.
$$

Sampled-token OPD 估计：

$$
\ell_{\mathrm{sample}}
=
\log p_t(\hat y_t)
-
\log q_t(\hat y_t),
$$

其中 `p_t` 可表示学生，`q_t` 表示教师。也可用 full-vocab 或 top-k KL 替代 sampled-token。

训练流程：训练前或训练中监控 teacher/student top-k overlap、overlap-token advantage、entropy gap；若 overlap 太低，先做 teacher-rollout cold start 或筛选更 teacher-aligned 的 prompt；再进入 OPD。

优点：给出 OPD 是否会成功的可观测诊断；强调“高分教师”不等于“可教教师”；解释长序列后缀指导质量下降问题。

局限：诊断依赖 logits；overlap 阈值不是通用常数；提高 overlap 的 cold start 会引入额外 off-policy 阶段。

与其他 OPD 方法的关系：它补充了 ExOPD 的“教师作为 reward”视角：reward 再好，如果学生和教师局部分布没有可学习交集，OPD 也可能失败。

### 08. Black-Box OPD

本地 PDF：[08-black-box-opd-2511.10643.pdf](papers/start-here/08-black-box-opd-2511.10643.pdf)  
原始链接：[https://arxiv.org/abs/2511.10643](https://arxiv.org/abs/2511.10643)

研究问题：当教师是闭源 API 或只返回文本时，无法做 token-level KL。如何仍然让学生在自己的 on-policy 输出上向教师学习？

核心方法：Generative Adversarial Distillation（GAD）。学生是 generator，判别器学习区分教师回答与学生回答。学生再用判别器分数作为 on-policy reward 更新。

关键公式。判别器可用 Bradley-Terry 风格目标：

$$
\min_\phi
\mathbb E
\left[
-
\log \sigma
\left(
D_\phi(x,y_T)-D_\phi(x,y_S)
\right)
\right],
$$

学生目标：

$$
\max_\theta
\mathbb E_{y_S\sim\pi_\theta(\cdot|x)}
\left[
D_\phi(x,y_S)
\right].
$$

训练流程：用教师 API 生成回答；学生 on-policy 生成回答；训练判别器偏好教师胜过学生；用在线判别器奖励通过 RL 更新学生；持续交替，避免固定 reward model 被学生 exploit。

优点：不需要 teacher logits；适合闭源模型、跨 tokenizer 教师；判别器在线更新可缓解 reward hacking。

局限：训练复杂度高于白盒 GKD；判别器质量决定上限；奖励不再是 token-level exact guidance，样本效率通常更低。

与其他 OPD 方法的关系：它把白盒 token KL 替换成 learned reward。概念上更接近 RLHF/RLAIF，但数据分布仍是学生 on-policy。

### 09. OPSD: On-Policy Self-Distillation

本地 PDF：[09-opsd-2601.18734.pdf](papers/start-here/09-opsd-2601.18734.pdf)  
原始链接：[https://arxiv.org/abs/2601.18734](https://arxiv.org/abs/2601.18734)

研究问题：没有外部强教师时，模型能否从自己的 privileged view 中学习推理能力？

核心方法：同一个模型同时扮演学生和教师。学生只看问题 `x`，教师额外看到参考推理或答案 `y^*`。学生先生成 rollout，教师在学生前缀上用 privileged context 给 token 分布指导。

关键公式：

$$
\mathcal L_{\mathrm{OPSD}}
=
\mathbb E_{(x,y^*)\sim \mathcal S,\,\hat y\sim p_S(\cdot|x)}
\left[
\sum_n
D
\left(
p_T(\cdot|x,y^*,\hat y_{<n})
\|
p_S(\cdot|x,\hat y_{<n})
\right)
\right].
$$

Sampled-token advantage：

$$
A_n
=
\log p_T(\hat y_n|x,y^*,\hat y_{<n})
-
\log p_S(\hat y_n|x,\hat y_{<n}).
$$

训练流程：构造带参考推理/答案的数据；学生在无答案条件下 rollout；同模型在有答案条件下评估同一前缀；用 full-vocab KL、reverse KL、forward KL 或 JSD 训练学生；可对 pointwise divergence 做 clipping。

优点：不需要外部教师；比 GRPO 等稀疏奖励更 token-efficient；privileged context 能提供密集推理提示。

局限：依赖高质量参考推理或答案；privileged teacher 可能在风格 token 上给过强信号；如果基础模型 ICL 能力弱，教师视角也不够强。

与其他 OPD 方法的关系：OPSD 是 self-distillation 版 GKD。它与 SDFT 的差异是 OPSD 更偏 reasoning privileged answer，SDFT 更偏 demonstration-conditioned continual learning。

### 10. SDFT: Self-Distillation Fine-Tuning

本地 PDF：[10-sdft-2601.19897.pdf](papers/start-here/10-sdft-2601.19897.pdf)  
原始链接：[https://arxiv.org/abs/2601.19897](https://arxiv.org/abs/2601.19897)

研究问题：持续学习中，SFT 容易过拟合新任务并遗忘旧能力。能否通过 demonstration-conditioned self-teacher 把新知识蒸进模型，同时保留泛化能力？

核心方法：同一模型在看到 expert demonstration `c` 时作为教师 `\pi(\cdot|x,c)`，学生 `\pi_\theta(\cdot|x)` 不看 demonstration。学生 on-policy 采样，再向 demonstration-conditioned teacher 做 reverse KL。实际训练中常使用 EMA teacher 提高稳定性。

关键公式：

$$
\mathcal L(\theta)
=
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x)
\|
\pi(\cdot|x,c)
\right)
=
\mathbb E_{y\sim\pi_\theta}
\left[
\log
\frac{\pi_\theta(y|x)}{\pi(y|x,c)}
\right].
$$

训练流程：准备示例或演示 `c`；学生在无演示条件下生成；教师在有演示条件下评估同一输出；用 reverse KL 更新学生；教师权重可用 EMA 维护。

优点：适合 continual learning；比直接 SFT 更不容易灾难性遗忘；能把 in-context 行为内化到权重。

局限：依赖模型本身能利用 demonstration；可能学到演示中的格式口癖；EMA teacher 引入额外工程复杂度。

与其他 OPD 方法的关系：SDFT 和 OPCD 都是在“有上下文教师、无上下文学生”之间蒸馏。SDFT 的上下文是 demonstration，OPCD 的上下文更一般，可以是知识、系统提示或经验。

### 11. OPCD: On-Policy Context Distillation

本地 PDF：[11-opcd-2602.12275.pdf](papers/start-here/11-opcd-2602.12275.pdf)  
原始链接：[https://arxiv.org/abs/2602.12275](https://arxiv.org/abs/2602.12275)

研究问题：长上下文、系统提示、检索知识或经验能提升模型输出，但部署时每次携带 context 昂贵且不稳定。能否把 context 的效果蒸进权重？

核心方法：学生 `\pi_\theta(\cdot|x)` 在没有 context 的条件下生成；教师 `\pi_T(\cdot|c,x)` 在有 context 的条件下评估学生前缀；训练学生在无 context 时匹配有 context 教师。

关键公式：

$$
\mathcal L_{\mathrm{OPCD}}(\theta)
=
\mathbb E_{(x,c),\,y\sim\pi_\theta(\cdot|x)}
\left[
\frac{1}{|y|}
\sum_t
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x,y_{<t})
\|
\pi_T(\cdot|c,x,y_{<t})
\right)
\right].
$$

训练流程：准备 prompt-context 对；学生无 context rollout；教师带 context 打分；通常在 top-k token 上近似 reverse KL；更新学生。

优点：把 prompt engineering、检索知识或经验内化到模型；减少推理时上下文长度；比 off-policy context distillation 更贴近学生真实行为。

局限：context 的知识会被压进权重，更新与遗忘需要管理；context 若不可靠，会固化错误；大规模 top-k KL 仍有计算成本。

与其他 OPD 方法的关系：OPCD 是 OEL 的核心 consolidation 机制，也是 SDFT 的泛化形式。

### 12. OEL: Online Experiential Learning

本地 PDF：[12-oel-2603.16856.pdf](papers/start-here/12-oel-2603.16856.pdf)  
原始链接：[https://arxiv.org/abs/2603.16856](https://arxiv.org/abs/2603.16856)

研究问题：真实部署中，环境交互发生在用户端，服务端可能拿不到环境、奖励或 verifier。如何从部署轨迹中持续学习经验？

核心方法：OEL 把 deployment trace 转成 experience，再用 OPCD 把 experience 蒸进模型。服务端不需要直接访问环境，也不依赖 reward model。

关键公式。经验抽取：

$$
e'_i
\sim
\pi_{\mathrm{extract}}(\cdot|\tau_i,e_{i-1}),
\quad
e_i=[e_{i-1};e'_i].
$$

经验 consolidation：

$$
\mathbb E_{y\sim\pi_\theta(\cdot|x)}
\sum_t
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x,y_{<t})
\|
\pi_T(\cdot|e,x,y_{<t})
\right).
$$

训练流程：用户端模型执行任务并收集多轮轨迹；服务端从轨迹中抽取和累积 experience；构造 partial rollout prefixes；用带 experience 的教师通过 OPCD 指导无 experience 学生；部署更新后的模型进入下一轮。

优点：适合无 reward、无 verifier、无服务端环境访问的持续学习；提取后的 experience 比原始轨迹更高效；可提升 token efficiency 并减少对长上下文依赖。

局限：experience extraction 质量是瓶颈；持续更新可能引入污染和遗忘；需要隐私、数据治理和版本回滚机制。

与其他 OPD 方法的关系：OEL 是 OPCD 在 deployment loop 中的系统化应用。

### 13. Qwen3 Technical Report

本地 PDF：[13-qwen3-2505.09388.pdf](papers/start-here/13-qwen3-2505.09388.pdf)  
原始链接：[https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

研究问题：如何把强大 teacher 的推理与通用能力转移到小模型，并在 thinking / non-thinking 多模式之间取得平衡？

核心方法：Qwen3 技术报告中的 OPD 是工业 post-training recipe 的一部分，而不是单独提出新公式。其小模型训练采用 strong-to-weak distillation，结合 off-policy 与 on-policy KD，从更强教师向较小 dense 模型迁移推理能力。

可用统一公式表示其中 on-policy KD 部分：

$$
\min_\theta
\mathbb E_{x,\,y\sim\pi_\theta(\cdot|x)}
\sum_t
D
\left(
\pi_{\mathrm{teacher}}(\cdot|x,y_{<t})
\|
\pi_\theta(\cdot|x,y_{<t})
\right).
$$

训练流程：先进行长 CoT cold start 和 reasoning RL；融合 thinking 与 non-thinking 模式；再对小模型进行 strong-to-weak distillation，其中包含教师轨迹上的 off-policy 学习和学生轨迹上的 on-policy 学习。

优点：证明 OPD 已进入主流工业训练栈；对小模型而言，强教师蒸馏在性能和效率上常比直接 RL 更划算。

局限：报告层面没有公开所有 OPD loss、采样比例和 teacher scheduling 细节；难以完全复现。

与其他 OPD 方法的关系：Qwen3 是 GKD/MiniLLM 思路的产品化强到弱应用。

### 14. DeepSeek-V4 Technical Report

本地 PDF：[14-deepseek-v4.pdf](papers/start-here/14-deepseek-v4.pdf)  
原始链接：[https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)

研究问题：多个领域专家通过 SFT + GRPO 获得不同能力后，如何整合到一个统一模型，而不被传统混合 RL 的不稳定性拖累？

核心方法：DeepSeek-V4 使用多教师 OPD 做 unified model consolidation。多个 domain expert 作为教师，学生在自己的轨迹上同时向多个专家学习。报告明确指出使用 full-vocabulary logit distillation 来降低 sampled-token KL 的高方差和不稳定。

关键公式：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\sum_{i=1}^{N}
w_i
D_{\mathrm{KL}}
\left(
\pi_\theta
\|
\pi_{E_i}
\right).
$$

训练流程：分别训练 math、coding、agent、instruction 等领域专家；统一学生 on-policy 采样；按 prompt/domain 调度专家；计算 full-vocab teacher logits；加权 KL 更新学生；通过 teacher weight offloading、last-layer hidden cache、logit reconstruction 和专用 kernel 控制成本。

优点：把 OPD 用作多专家能力合并工具；full-vocab 蒸馏比 sampled-token 更稳；工程部分展示了大规模 OPD 的真实瓶颈在 teacher inference/logit 计算。

局限：需要访问多个专家模型的 logits；系统工程复杂；专家冲突和权重 `w_i` 需要策略设计。

与其他 OPD 方法的关系：DeepSeek-V4 是 ExOPD/MOPD 思路的工业版本，强调多教师、full-vocab、调度与计算优化。

### 15. MiMo-V2-Flash

本地 PDF：[15-mimo-v2-flash-2601.02780.pdf](papers/start-here/15-mimo-v2-flash-2601.02780.pdf)  
原始链接：[https://arxiv.org/abs/2601.02780](https://arxiv.org/abs/2601.02780)

研究问题：如何把不同领域 teacher 的专长整合到一个高效模型中，同时允许 outcome reward 与 teacher token reward 协同？

核心方法：Multi-Teacher On-Policy Distillation（MOPD）。学生 on-policy 采样，按样本所属领域选择 domain teacher，使用 teacher/student log-prob ratio 形成 token-level advantage，并可与 outcome reward model 的 advantage 混合。

关键公式。teacher KL reward advantage：

$$
\hat A_{\mathrm{MOPD},t}
=
\operatorname{sg}
\left[
\log
\frac{
\pi_{\mathrm{domain}}^x(y_t|x,y_{<t})
}{
\pi_\theta(y_t|x,y_{<t})
}
\right].
$$

surrogate loss：

$$
\mathcal L_{\mathrm{MOPD}}(\theta)
=
-
\mathbb E_{x\sim\mathcal D,\,y\sim\mu_\theta}
\left[
\frac{1}{|y|}
\sum_t
w_t
\hat A_{\mathrm{MOPD},t}
\log\pi_\theta(y_t|x,y_{<t})
\right].
$$

与 ORM 混合：

$$
\hat A_t
=
\hat A_{\mathrm{MOPD},t}
+
\alpha \hat A_{\mathrm{ORM}}.
$$

训练流程：SFT 得到基础模型；训练或准备领域专家；学生采样时选择 domain teacher；用重要性权重 `w_t` 截断训练/采样策略概率比；将 teacher advantage 与 outcome reward 组合；允许 teacher-student co-evolution。

优点：适合多领域能力整合；token 级教师信号和 outcome reward 可互补；重要性截断提升稳定性。

局限：teacher 选择、权重和 reward 混合较复杂；需要多教师维护成本；sampled-token advantage 仍有方差问题。

与其他 OPD 方法的关系：MOPD 继承 MiniLLM token advantage，吸收 ExOPD 的 dense reward 视角，并与 DeepSeek-V4 的多教师整合目标高度相近。

### 16. GLM-5 Technical Report

本地 PDF：[16-glm-5-2602.15763.pdf](papers/start-here/16-glm-5-2602.15763.pdf)  
原始链接：[https://arxiv.org/abs/2602.15763](https://arxiv.org/abs/2602.15763)

研究问题：多阶段 post-training 会在新阶段提升某些能力，同时遗忘早期阶段学到的能力。如何在最终模型中恢复 reasoning、agentic、general 等不同阶段能力？

核心方法：On-Policy Cross-Stage Distillation。GLM-5 把前面不同训练阶段的 checkpoint 当作教师，让最终学生在自己的 rollout 上向这些阶段教师学习，恢复早期阶段的能力。

统一公式：

$$
\mathcal L_{\mathrm{cross-stage}}
=
\sum_s
\alpha_s
\mathbb E_{x\sim\mathcal D_s,\,y\sim\pi_\theta}
\sum_t
D
\left(
\pi_{\theta}(\cdot|x,y_{<t})
\|
\pi_{\mathrm{stage}\,s}(\cdot|x,y_{<t})
\right).
$$

训练流程：保留 SFT、Reasoning RL、Agentic RL、General RL 等阶段 checkpoint；按阶段任务采样 prompt；当前最终学生 on-policy 生成；对应阶段教师在同一前缀上给指导；把能力重新蒸回最终模型。

优点：把 OPD 用于生命周期内的能力防遗忘；不要求外部更强教师；适合复杂 post-training pipeline 的最终收敛。

局限：报告没有公开完整训练细节；阶段教师之间可能冲突；需要保存和调度多个 checkpoint。

与其他 OPD 方法的关系：GLM-5 是 self/multi-teacher OPD 的工业生命周期版本。与 DeepSeek-V4 的领域专家不同，GLM-5 的教师主要来自训练时间轴上的不同阶段。

## 4. 横向对比

| 类别 | 教师访问 | 代表方法 | 核心信号 | 主要优点 | 主要风险 |
|---|---|---|---|---|---|
| White-box OPD | 教师 logits/log-probs | MiniLLM、GKD、Revisiting、Entropy-Aware、Rethinking | token KL、top-k KL、sampled-token advantage | 稠密、样本效率高、可诊断 | 计算贵，教师偏离前缀可能不可靠 |
| Black-box OPD | 只有教师文本/API | Black-Box OPD | 判别器 reward、偏好信号 | 适合闭源教师和跨 tokenizer | reward 学习难，稳定性依赖判别器 |
| Self-distillation | 同模型不同条件 | OPSD、SDFT | privileged context KL | 无需外部教师，成本低 | 依赖模型 ICL/自解释能力 |
| Context distillation | 带 context 教师 | OPCD、OEL | context-conditioned KL | 把上下文、经验蒸进权重 | 容易固化错误知识，需管理更新 |
| Industrial recipes | 多教师/多阶段 | Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 | full-vocab KL、multi-teacher KL、token reward | 真实生产可扩展，能整合多能力 | teacher scheduling、系统成本、复现难 |

## 5. 实践路线

### 5.1 入门实现：MiniLLM / GKD

先实现一个 GKD 风格训练循环，而不是一开始就实现 full policy-gradient MiniLLM。最小闭环是：

1. 从 prompt 池采样 `x`。
2. 学生 `\pi_\theta` 生成 `y`。
3. 教师在 `(x,y_{<t})` 上输出 logits。
4. 对每个 token 计算 KL 或 JSD。
5. 对学生参数反向传播，但不穿过采样过程。

最小目标：

$$
\mathcal L
=
\frac{1}{T}
\sum_t
D_{\mathrm{KL}}
\left(
\pi_T(\cdot|c_t)
\|
\pi_\theta(\cdot|c_t)
\right).
$$

如果算 full-vocab 太贵，先用 teacher top-k 或 student top-k 近似，但要记录 top-k 策略，因为它会改变 loss 性质。

### 5.2 稳定性修正：support、entropy、overlap

当基础 GKD 能跑通后，再加入三类监控：

| 监控 | 目的 | 对应论文 |
|---|---|---|
| teacher/student top-k overlap | 判断教师是否“可教” | Rethinking OPD |
| teacher entropy | 判断是否需要保留多样性 | Entropy-Aware OPD |
| sampled-token vs top-k/full-vocab gap | 判断 estimator 是否不稳定 | Revisiting OPD |

若 overlap 低，先用 teacher rollout 做 warmup，或筛选更 aligned prompt。若 entropy 高且多样性下降，加入 forward KL。若 sampled-token 方差大，切换到 top-k support matching 或 full-vocab KL。

### 5.3 无 logits：Black-Box OPD

如果只能访问 API teacher，直接做 token KL 不可行。可用 GAD 路线：

1. 教师 API 和学生分别对同一 prompt 生成回答。
2. 判别器学习 `teacher > student`。
3. 学生用判别器分数做 on-policy RL。
4. 判别器持续在线更新，避免学生 exploit 固定 reward。

该路线更像 RL 系统，优先关注 reward hacking、判别器过拟合和 teacher data 刷新。

### 5.4 自蒸馏：privileged context

若没有外部教师但有答案、示例、检索上下文或经验，优先考虑 OPSD/SDFT/OPCD：

$$
\text{teacher input}=x+z,\quad
\text{student input}=x.
$$

让学生在无 `z` 的条件下生成，再让带 `z` 的 teacher 评价相同前缀。这个方向适合把 prompt、context、reference reasoning 或 demonstration 内化到权重。

### 5.5 工业多阶段配方

工业 OPD 的重点不只是 loss，而是管线：

| 阶段 | 关键决策 |
|---|---|
| SFT / cold start | 是否先提高 student-teacher overlap |
| RL / expert training | 是否为每个领域训练 expert |
| OPD consolidation | full-vocab、top-k 还是 sampled-token；单教师还是多教师 |
| 调度与系统 | teacher 何时加载、logits 是否缓存、权重如何 offload |
| 评估 | 单任务峰值、跨任务平均、遗忘、pass@k、token efficiency |

DeepSeek-V4 更强调多领域专家合并；MiMo-V2-Flash 强调 MOPD 与 outcome reward 混合；GLM-5 强调跨训练阶段恢复能力；Qwen3 强调 strong-to-weak 小模型蒸馏效率。

## 6. 读论文时的抽取模板

每读一篇 OPD 论文，都建议填下面 8 项：

| 字段 | 需要回答的问题 |
|---|---|
| objective / loss | 最小化 KL、JSD、sampled-token PG，还是最大化 reward？ |
| sampling policy | 轨迹来自学生、教师、数据集，还是混合？是否有温度/top-p？ |
| teacher access | 能否拿 logits？只能拿文本？是否同模型 privileged context？ |
| token-level signal | full-vocab、top-k、sampled token、sequence reward？ |
| stability mechanism | clipping、support matching、entropy gate、importance weight、EMA teacher？ |
| teacher reliability | 教师在学生偏离前缀上是否可靠？如何诊断？ |
| experiment conclusion | 提升来自 on-policy、teacher quality、loss 形式还是训练管线？ |
| implementation bottleneck | teacher inference、logit 存储、分布式采样、reward 模型，哪个最贵？ |

## 7. 参考资料索引

| 优先级 | 顺序 | 资料 | 本地 PDF | 原始 URL | 阅读目的 |
|---|---:|---|---|---|---|
| P0 | 01 | OPD Survey | [PDF](papers/start-here/01-opd-survey-2604.00626.pdf) | [arXiv](https://arxiv.org/abs/2604.00626) | 建立全局分类 |
| P0 | 02 | MiniLLM | [PDF](papers/start-here/02-minillm-2306.08543.pdf) | [arXiv](https://arxiv.org/abs/2306.08543) | reverse KL 与早期 OPD |
| P0 | 03 | GKD | [PDF](papers/start-here/03-gkd-2306.13649.pdf) | [arXiv](https://arxiv.org/abs/2306.13649) | on/off-policy 混合目标 |
| P0 | 04 | ExOPD | [PDF](papers/start-here/04-exopd-2602.12125.pdf) | [arXiv](https://arxiv.org/abs/2602.12125) | OPD 作为 dense RL |
| P0 | blog | Thinking Machines OPD | 不适用 | [Blog](https://thinkingmachines.ai/blog/on-policy-distillation/) | 工程直觉 |
| P1 | 05 | Revisiting OPD | [PDF](papers/start-here/05-revisiting-opd-2603.25562.pdf) | [arXiv](https://arxiv.org/abs/2603.25562) | 稳定性与 support matching |
| P1 | 06 | Entropy-Aware OPD | [PDF](papers/start-here/06-entropy-aware-opd-2603.07079.pdf) | [arXiv](https://arxiv.org/abs/2603.07079) | 高熵多样性 |
| P1 | 07 | Rethinking OPD | [PDF](papers/start-here/07-rethinking-opd-2604.13016.pdf) | [arXiv](https://arxiv.org/abs/2604.13016) | 成功条件与失败诊断 |
| P1 | 08 | Black-Box OPD | [PDF](papers/start-here/08-black-box-opd-2511.10643.pdf) | [arXiv](https://arxiv.org/abs/2511.10643) | 无 logits 教师 |
| P1 | 09 | OPSD | [PDF](papers/start-here/09-opsd-2601.18734.pdf) | [arXiv](https://arxiv.org/abs/2601.18734) | privileged reasoning 自蒸馏 |
| P1 | 10 | SDFT | [PDF](papers/start-here/10-sdft-2601.19897.pdf) | [arXiv](https://arxiv.org/abs/2601.19897) | demonstration 自蒸馏 |
| P1 | 11 | OPCD | [PDF](papers/start-here/11-opcd-2602.12275.pdf) | [arXiv](https://arxiv.org/abs/2602.12275) | 上下文蒸馏 |
| P1 | 12 | OEL | [PDF](papers/start-here/12-oel-2603.16856.pdf) | [arXiv](https://arxiv.org/abs/2603.16856) | 部署经验学习 |
| P2 | 13 | Qwen3 | [PDF](papers/start-here/13-qwen3-2505.09388.pdf) | [arXiv](https://arxiv.org/abs/2505.09388) | 强到弱工业蒸馏 |
| P2 | 14 | DeepSeek-V4 | [PDF](papers/start-here/14-deepseek-v4.pdf) | [Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | 多专家 OPD 整合 |
| P2 | 15 | MiMo-V2-Flash | [PDF](papers/start-here/15-mimo-v2-flash-2601.02780.pdf) | [arXiv](https://arxiv.org/abs/2601.02780) | MOPD + outcome reward |
| P2 | 16 | GLM-5 | [PDF](papers/start-here/16-glm-5-2602.15763.pdf) | [arXiv](https://arxiv.org/abs/2602.15763) | 跨阶段 OPD 防遗忘 |

## 8. 最短学习顺序

如果只想用 2 天建立可实现的 OPD 认知：

1. 读 Survey 的 taxonomy，建立术语。
2. 精读 MiniLLM 和 GKD，手写 student rollout + teacher KL 训练循环。
3. 读 ExOPD，理解 OPD 与 RL 的等价关系。
4. 读 Revisiting、Entropy-Aware、Rethinking，补齐稳定性和失败诊断。
5. 按需要选 Black-Box、OPSD/SDFT、OPCD/OEL。
6. 最后读 Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5，把单一算法放回完整 post-training pipeline。

完成这条路线后，应能回答三个工程问题：是否有 teacher logits；学生和教师是否足够可教；最终目标是压缩、整合、蒸上下文，还是替代稀疏奖励 RL。
