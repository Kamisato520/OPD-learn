# ⚗️ Awesome On-Policy Distillation

<p align="center">
  <a href="https://awesome.re"><img src="https://img.shields.io/badge/Awesome-%E2%9A%97%EF%B8%8F_On--Policy_Distillation-000000?style=for-the-badge&labelColor=000000" alt="Awesome On-Policy Distillation"></a>
</p>

<p align="center">
  <!-- entry-count-start --><a href="#contents"><img src="https://img.shields.io/badge/Entries-229-000000?style=for-the-badge&labelColor=000000" alt="Entries"></a><!-- entry-count-end -->
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/stargazers"><img src="https://img.shields.io/github/stars/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Stars&labelColor=000000&color=000000" alt="GitHub Stars"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/network/members"><img src="https://img.shields.io/github/forks/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Forks&labelColor=000000&color=000000" alt="GitHub Forks"></a>
  <a href="https://github.com/chrisliu298/awesome-on-policy-distillation/commits"><img src="https://img.shields.io/github/last-commit/chrisliu298/awesome-on-policy-distillation?style=for-the-badge&logo=github&logoColor=white&label=Last%20Commit&labelColor=000000&color=000000" alt="Last Commit"></a>
</p>

A curated collection of papers, technical reports, frameworks, and tools for on-policy distillation (OPD) of large language models.

> **On-policy distillation** trains a student on samples from its own evolving policy, while a teacher (external, privileged, or self-conditioned) provides dense supervision on those same samples.

On-policy distillation (OPD) trains a student on trajectories sampled from its own policy while a teacher scores the student-visited prefixes with dense token-level guidance. This on-policy data collection reduces the train-inference distribution gap that affects off-policy KD/SFT on fixed traces. Depending on the estimator, OPD looks like GKD on student rollouts or policy-gradient/RL with teacher-defined per-token KL/log-prob rewards, making the natural contrast sparse outcome-reward RL rather than RL as a whole. As of 2026, OPD is a standard post-training primitive at Alibaba (Qwen3), DeepSeek (V4), Xiaomi (MiMo), Zhipu (GLM-5), NVIDIA (Nemotron-Cascade 2), and others.

**Shipping today?** Jump to [Frameworks and Implementations](#frameworks-and-implementations). **New to OPD?** Read [Start Here](#start-here).

## Contents

- [Start Here](#start-here)
- [Surveys and Essays](#surveys-and-essays)
- [Core OPD Papers](#core-opd-papers)
  - [Foundations](#foundations)
  - [Gap-Bridging](#gap-bridging)
  - [Stability and Objective Design](#stability-and-objective-design)
  - [Self-Distillation](#self-distillation)
  - [Context and Experience Internalization](#context-and-experience-internalization)
  - [Efficiency, Systems, and Privacy Variants](#efficiency-systems-and-privacy-variants)
- [Taxonomy](#taxonomy)
- [Adjacent and Enabling Work](#adjacent-and-enabling-work)
- [Domain Extensions](#domain-extensions)
- [Technical Reports and Industrial Recipes](#technical-reports-and-industrial-recipes)
- [Frameworks and Implementations](#frameworks-and-implementations)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Citation](#citation)

## Start Here

A fast path through the field:

1. **Survey.** [OPD Survey](https://arxiv.org/abs/2604.00626) — taxonomy, methods, and open problems in one place.
2. **Foundations.** [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649), and [ExOPD](https://arxiv.org/abs/2602.12125) — the core student-rollout plus teacher-supervision loop, including its dense KL-constrained RL framing.
3. **Practical intuition.** [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/) — the clearest end-to-end explanation of why and when OPD applies.
4. **When OPD works and when it breaks.** [Revisiting OPD](https://arxiv.org/abs/2603.25562), [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079), and [Rethinking OPD](https://arxiv.org/abs/2604.13016) — failure modes (instability, diversity collapse, tokenizer mismatch) and success conditions (compatible thinking patterns, novel teacher capability).
5. **No teacher logits.** [Black-Box OPD](https://arxiv.org/abs/2511.10643) — discriminator-based reward when the teacher is API-only.
6. **No teacher at all.** [OPSD](https://arxiv.org/abs/2601.18734) and [SDFT](https://arxiv.org/abs/2601.19897) — same model as student and self-teacher.
7. **Context and experience.** [OPCD](https://arxiv.org/abs/2602.12275) and [OEL](https://arxiv.org/abs/2603.16856) — distill prompts and deployment traces into weights.
8. **Industrial recipes.** [Qwen3](https://arxiv.org/abs/2505.09388), [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780), [GLM-5](https://arxiv.org/abs/2602.15763) — how labs ship OPD in production.

**Key decision:** access to teacher logits? Yes → white-box (GKD, Veto, Entropy-Aware OPD). No → black-box (GAD, OVD) or self-distillation (OPSD, SDFT).

## Surveys and Essays

### Surveys and Position Papers

- [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) *(2026)* — First dedicated OPD survey; organizes methods by feedback signal, teacher access mode, and loss scope.
- [Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation](https://arxiv.org/abs/2605.22731) *(2026)* — Reframes SFT/RL/OPD by training-state source rather than loss, explaining why OPD's student-sampled states beat a degraded teacher.

### Essays, Blog Posts, and Walkthroughs

- [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) *(2025)* — Best single-article introduction. Covers concepts, intuition, and practical use cases.
- [Unlocking On-Policy Distillation for Any Model Family (GOLD)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) *(2025)* — Cross-tokenizer OPD walkthrough with TRL code.
- [Distilling 100B+ Models 40x Faster with TRL](https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer) *(2026)* — HF engineering walkthrough of TRL's `DistillationTrainer` scaling tricks; ~40× speedup, validated on Qwen3-235B → Qwen3-4B math.
- [Multi-Teacher On-Policy Distillation: A New Post-Training Primitive](https://yumoxu.notion.site/multi-teacher-on-policy-distillation) *(2026)* — Yumo Xu surveys MOPD as a post-training primitive across MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2, DeepSeek-V4.
- [On-Policy Distillation: Theory & Practice in Model Merging](https://www.notion.so/On-Policy-Distillation-Theory-Practice-in-Model-Merging-2f44795a3e8b801cbedee2c96a23c788) *(2026)* — ByteDance Seed framing OPD as entropy-regularized RL; cross-tokenizer pitfalls and reward hacking in agent merging.
- [On SFT, RL, and on-policy distillation](https://x.com/willccbb/status/2050038277454143918) *(2026)* — Will Brown's essay on OPD via SFT-vs-RL compounding and gradient geometry; pointers toward an optimal teacher.
- [SFT, RL, and OPD Through a Distributional Lens](https://x.com/nrehiew_/status/2053482349300797526) *(2026)* — wh's distributional-geometry framing; experiment shows OPD students from SFT and RL teachers converge and forget less.
- [On Policy Self Distillation](https://x.com/ar0cket1/status/2054108160450064571) *(2026)* — KL-geometry study showing OPSD inverts OPD's per-token sign and suffers larger KL shocks that GEPA hint evolution roughly halves.
- [What Apple found out about On-Policy Distillation](https://x.com/neural_avb/status/2054585001757614172) *(2026)* — AVB's tutorial-style breakdown of "Unmasking OPD"; training-free gradient-alignment for predicting student-teacher fit.
- [OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践 / OPD Deep Dive: From Mathematical Derivation to DeepSeek V4, SWIFT, and verl Practice](https://zhuanlan.zhihu.com/p/2033212181823608430) *(2026)* — Chinese-language Zhihu deep-dive deriving OPD's sequence- and token-level reverse-KL; maps variants to MiniLLM, GKD, verl, DeepSeek V4.
- [重温 On-Policy Distillation / Revisiting On-Policy Distillation](https://yph22.github.io/files/notes/OPD.pdf) *(2026)* — Chinese-language notes deriving OPD as both a SeqKD student-rollout mirror and RL with token-level teacher supervision.
- [The Imitation Game: State of Policy Distillation in Language Model training](https://hackmd.io/@l_WDq7lkQq29Pz-KD1JPNA/r1oNsX9Jfl) *(2026)* — Long-form OPD/OPSD survey with a four-axis failure-modes taxonomy; argues hybrid OPSD and cross-tokenizer OPD as the highest-leverage open problems.

## Core OPD Papers

The papers that define on-policy distillation for LLMs.

**Scope rule:** A paper belongs here if its primary contribution is a new component of the OPD training loop itself — an objective, divergence formulation, stability fix, teacher access-mode variant, self-distillation variant, context-internalization mechanism, or systems/efficiency/privacy constraint applied to that loop — with student rollouts central to the learning signal, evaluated on LLM text generation or reasoning. **Operational test:** if removing the OPD-loop component leaves a working contribution (a working RL recipe, preference method, or KD baseline), the OPD piece is auxiliary → Adjacent. Papers that *enable* OPD (cross-tokenizer alignment, calibration), *compose with* OPD as one component of a larger RL/preference structure, or apply OPD to non-text-reasoning substrates live in [Adjacent and Enabling Work](#adjacent-and-enabling-work) or [Domain Extensions](#domain-extensions).

### Foundations

- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) *(2023)* — Reverse-KL framing for generative LMs; the paper that named the field.
- [GKD: On-Policy Distillation of Language Models — Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) *(2023)* — Unifying formulation spanning on-/off-policy mixtures with flexible divergences.

### Gap-Bridging

- [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) *(2024)* — Interleaved teacher/student sampling mitigates poor student rollout quality.
- [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643) *(2025)* — GAD: discriminator-based reward on student rollouts; no teacher logits required.
- [SOD: Step-wise On-policy Distillation for Small Language Model Agents](https://arxiv.org/abs/2605.07725) *(2026)* — Reweights teacher guidance by step-level divergence to avoid tool-induced cascade drift.
- [MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate](https://arxiv.org/abs/2605.01347) *(2026)* — Multi-agent debate consensus as the OPD teacher; extends to agentic tasks via step-level sampling.
- [ROPD: Rubric-based On-policy Distillation](https://arxiv.org/abs/2605.07396) *(2026)* — Black-box OPD using prompt-specific rubrics distilled from teacher-student contrasts to score rollouts.

### Stability and Objective Design

- [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898) *(2024)* — Skew-KL divergence with adaptive off-policy use of student-generated outputs; foundational OPD objective formulation (ICML 2024).
- [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://arxiv.org/abs/2503.07067) *(2025)* — Contrastive extension of skew-KL; student-generated outputs collected per epoch.
- [Veto: Stable On-Policy Distillation through Adaptive Target Reformulation](https://arxiv.org/abs/2601.07155) *(2026)* — Intermediate target distribution in logit space stabilizes training.
- [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079) *(2026)* — Forward-KL on high-entropy teacher tokens preserves output diversity.
- [ExOPD: Learning beyond Teacher via Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) *(2026)* — Casts OPD as dense KL-constrained RL; reward scaling enables teacher-surpassing behavior.
- [REOPOLD: Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) *(2026)* — Relaxes imitation with reward clipping, entropy-based dynamic sampling, and explore-to-refine training.
- [PACED: Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) *(2026)* — Pass-rate weighting focuses learning on the student's competence frontier.
- [Revisiting On-Policy Distillation — Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) *(2026)* — Truncated reverse-KL with teacher top-K support matching; fixes imbalanced signals and tokenizer mismatch.
- [Rethinking On-Policy Distillation — Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016) *(2026)* — Identifies compatible thinking patterns and novel teacher capability as OPD success conditions.
- [The Illusion of Certainty — Decoupling Capability and Calibration in OPD](https://arxiv.org/abs/2604.16830) *(2026)* — Diagnoses OPD-induced overconfidence; CaOPD replaces confidence targets with student-grounded empirical success rates.
- [Demystifying OPD — Length Inflation and Stabilization Strategies](https://arxiv.org/abs/2604.08527) *(2026)* — Repetition-driven length inflation in iterative OPD; Stable-OPD adds divergence constraints and a rollout-mixture anchor.
- [SCOPE: Signal-Calibrated On-Policy Distillation with Dual-Path Adaptive Weighting](https://arxiv.org/abs/2604.10688) *(2026)* — Routes correct student rollouts to student-PPL-weighted MLE and incorrect to teacher-PPL-weighted KL; dual-path OPD loss design.
- [HPD: Hybrid Policy Distillation for LLMs](https://arxiv.org/abs/2604.20244) *(2026)* — Unified reweighted-log-likelihood framework combining forward/reverse KL with off-policy and on-policy sampling.
- [Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe](https://arxiv.org/abs/2605.03677) *(2026)* — Offline difficulty-aware and online correctness-aware data balancing with outcome-guided margin calibration.
- [AOPD: Asymmetric On-Policy Distillation](https://arxiv.org/abs/2605.06387) *(2026)* — Replaces ineffective negative reinforcement with localized teacher-distribution matching in non-positive advantage regions.
- [vOPD: On-Policy Distillation with a Control Variate Baseline](https://arxiv.org/abs/2605.07865) *(2026)* — Closed-form per-token reverse-KL value baseline; unbiased lower-variance single-sample estimator with no extra critic.
- [Unmasking On-Policy Distillation — Where It Helps, Where It Hurts, and Why](https://arxiv.org/abs/2605.10889) *(2026)* — Training-free gradient-alignment diagnostic; best teacher flips with student capacity and task; wrong demos hurt self-distillation except on hard math.
- [The Many Faces of On-Policy Distillation — Pitfalls, Mechanisms, and Fixes](https://arxiv.org/abs/2605.11182) *(2026)* — Names three failure modes (student-prefix teacher-state mismatch, biased Top-K gradients, PI-free OPSD aggregation) and three stabilizers (stop-grad Top-K KL, RLVR teachers, SFT-stabilized students).
- [Rock Tokens — Deciphering High-Loss Tokens in On-Policy Distillation](https://arxiv.org/abs/2605.09253) *(2026)* — High-loss tokens (up to 18%) persist after apparent convergence; masking them streamlines alignment. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/YuxuanJiang1/Rock-Token)
- [BRTS: On-Policy Distillation with Best-of-N Teacher Rollout Selection](https://arxiv.org/abs/2605.09725) *(2026)* — Auxiliary teacher-context branch alongside standard OPD; selects best-of-N teacher rollouts by correctness then student-alignment. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/BWGZK-keke/BRTS)
- [Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation](https://arxiv.org/abs/2605.13643) *(2026)* — Dynamic release rule truncates dense supervision where the teacher's local margin collapses; counters suffix degradation in strong-to-weak OPD.
- [Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for LLM Post-Training](https://arxiv.org/abs/2605.12483) *(2026)* — Sparse-to-dense post-training workflow framing OPD as the dense teacher-induced reward between GRPO stages.
- [The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs](https://arxiv.org/abs/2605.08737) *(2026)* — Reward-extrapolation OPD collapses past a clip threshold on near-deterministic structured outputs, mapping where teacher-surpassing reward scaling stops working.

### Self-Distillation

- [OPSD: Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) *(2026)* — Single model as both teacher and student via privileged information; no external teacher.
- [SDFT: Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) *(2026)* — Demonstration-conditioned self-teaching for continual learning with less forgetting.
- [SDPO: Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) *(2026)* — Converts textual feedback into dense self-teacher signals for RL-like training.
- [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](https://arxiv.org/abs/2603.24472) *(2026)* — Traces failures to suppression of epistemic verbalization; task coverage determines whether conciseness helps.
- [OPSDC: On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433) *(2026)* — Compresses verbose reasoning using concise privileged self-teachers.
- [GATES: Self-Distillation under Privileged Context with Consensus Gating](https://arxiv.org/abs/2602.20574) *(2026)* — Consensus-gated asymmetric-context self-distillation without labels or rewards.
- [HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation](https://arxiv.org/abs/2603.23871) *(2026)* — Privileged self-distillation on cliff prompts where RL gradients vanish; recovers KL-regularized optimal policy.
- [RLSD: Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) *(2026)* — Self-distillation as token-level credit assignment within GRPO; OPSD-style matching leaks privileged information.
- [SDZero: Self-Revision Turns Binary Rewards into Dense Supervision](https://arxiv.org/abs/2604.12002) *(2026)* — Generator-reviser dual roles; reviser converts binary feedback into token-level supervision with no external teacher.
- [OPSDL: On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535) *(2026)* — Short-context distribution of the same model as co-evolving reverse-KL teacher under long context.
- [PBSD: Preference-Based Self-Distillation — Beyond KL Matching via Reward Regularization](https://arxiv.org/abs/2605.05040) *(2026)* — DPO-style preference learning between context-augmented teacher positives and on-policy student negatives.
- [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) *(2026)* — Unifies self-distillation across supervision reliability, representation alignment, and training stability. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ahren09/UniSD)
- [OPSD Compresses What RLVR Teaches — A Post-RL Compaction Stage](https://arxiv.org/abs/2605.06188) *(2026)* — Correct-only OPSD preserves accuracy and shortens responses; proposes SFT → RLVR → OPSD as post-RL compaction.
- [ATESD: Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning](https://arxiv.org/abs/2605.11458) *(2026)* — Treats teacher reveal ratio as a learnable control variable via Beta-policy controller with discounted learning-progress reward.
- [OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering](https://arxiv.org/abs/2605.12400) *(2026)* — Contrasts averaged teacher logits over correct vs. incorrect rollouts to form outcome-guided steering on anchor logits.
- [RLRT: Rebellious Student — Reversing Teacher Signals for Reasoning Exploration](https://arxiv.org/abs/2605.10781) *(2026)* — Upweights student tokens that diverged from teacher but still succeeded as a "valuable exploration" signal added to GRPO; +8.9% average across six math benchmarks.
- [EGRSD: Respecting Self-Uncertainty in On-Policy Self-Distillation for Efficient LLM Reasoning](https://arxiv.org/abs/2605.13255) *(2026)* — Teacher-entropy confidence gate over RLSD's direction-magnitude signal; causal-lookahead variant preserves transient pivot tokens (COLM 2026).
- [CREDIT: From Generic Correlation to Input-Specific Credit in On-Policy Self Distillation](https://arxiv.org/abs/2605.11613) *(2026)* — Recasts the self-distillation token reward as Bayesian filtering; batch-contrastive teacher baseline strips input-generic shortcuts (NeurIPS 2026).
- [OPHSD: Training with Harnesses — On-Policy Harness Self-Distillation for Complex Reasoning](https://arxiv.org/abs/2605.08741) *(2026)* — Generalizes self-distillation privileged context from a static variable (reference solution, environment trace) to a harness-driven workflow (draft-verify, plan-solve); harness is a removable training scaffold, +10.83% over OPSD on HMMT25. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation)
- [MixSD: Mixed Contextual Self-Distillation for Knowledge Injection](https://arxiv.org/abs/2605.16865) *(2026)* — Per-token Bernoulli mix of fact-conditioned and naive-conditioned base-model samples; replaces SFT for knowledge injection without collapsing held-out capability.
- [AntiSD: Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information](https://arxiv.org/abs/2605.11609) *(2026)* — Identifies the OPSD token reward as a PMI that suppresses deliberation tokens, then reverses its sign under an entropy-triggered gate. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/FloyedShen/AntiSD)
- [TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment](https://arxiv.org/abs/2605.10194) *(2026)* — Routes self-distillation KL only to annotator-marked spans to cure the all-token "distillation tax" of SDPO/SRPO.
- [AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals](https://arxiv.org/abs/2605.20643) *(2026)* — Multi-view privileged self-distillation that gates teacher-specific residuals so they can adjust update magnitude but cannot reverse the cross-view consensus direction. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/duykhuongnguyen/AVSD)
- [VPD: Learning from Language Feedback via Variational Policy Distillation](https://arxiv.org/abs/2605.15113) *(2026)* — Variational-EM self-distillation refines a feedback-conditioned self-teacher in the E-step before distilling it back via token-level KL in the M-step.
- [RMSD: Bringing Capabilities in Distribution via Relevance-Masked Self-Distillation](https://www.appliedcompute.com/research/relevance-masked-self-distillation) *(2026)* — Applied Compute's OPSD variant masking the reverse-KL loss to LLM-judge-selected behavior-relevant tokens; preserves capabilities where SFT collapses.
- [SPD: Self-Policy Distillation via Capability-Selective Subspace Projection](https://arxiv.org/abs/2605.22675) *(2026)* — Decode-time KV-subspace projection biases self-rollout generation toward capability-relevant directions, then LoRA-SFTs on those rollouts without any external verifier or teacher.
- [Multilingual Safety Alignment via Self-Distillation](https://arxiv.org/abs/2605.02971) *(2026)* — Same-model OPSD transfers English safety reasoning to low-resource languages without any response data.
- [COPSD: Crosslingual On-Policy Self-Distillation for Multilingual Reasoning](https://arxiv.org/abs/2605.09548) *(2026)* — Uses English translations and reference solutions as privileged teacher context for low-resource multilingual reasoning OPSD.

### Context and Experience Internalization

- [OPCD: On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275) *(2026)* — Context-conditioned teacher on student rollouts; distills system prompts and experiential knowledge.
- [OEL: Online Experiential Learning for Language Models](https://arxiv.org/abs/2603.16856) *(2026)* — Deployment loop using OPCD for consolidating interaction traces into weights.
- [Aligning Language Models from User Interactions](https://arxiv.org/abs/2603.12273) *(2026)* — Hindsight self-distillation from user follow-ups; same model conditioned on the follow-up serves as the teacher.

### Efficiency, Systems, and Privacy Variants

- [Prefix OPD: Fast and Effective On-policy Distillation from Reasoning Prefixes](https://arxiv.org/abs/2602.15260) *(2026)* — Distills only reasoning prefixes, cutting training FLOPs 2×-47×.
- [OVD: On-policy Verbal Distillation](https://arxiv.org/abs/2601.21968) *(2026)* — Trajectory-level verbal scoring instead of token-level logit matching; relaxes alignment requirements.
- [pi-Distill: Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942) *(2026)* — Training-time privileged information in agentic settings where only actions are observable.
- [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline OPD](https://arxiv.org/abs/2604.13010) *(2026)* — Precomputes teacher log-probs once over SFT rollouts; 4× speedup via teacher-consistency condition.
- [DP-OPD: Differentially Private On-Policy Distillation for Language Models](https://arxiv.org/abs/2604.04461) *(2026)* — Student-rollout OPD with DP-SGD on student updates; first OPD recipe with sample-level differential privacy.
- [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) *(2026)* — Selective training on high-entropy and confidently-wrong low-entropy tokens; matches full-token baselines at lower memory.
- [Nitrobrew: Communication- and Memory-Efficient On-Policy Distillation](https://blog.tilderesearch.com/blog/nitrobrew) *(2026)* — Hidden-state teacher→student transport plus tile-wise online divergence kernel; 1.5-3× throughput.
- [NPD: Near-Policy Distillation via Asynchronous Generation and Selective Packing](https://arxiv.org/abs/2605.05940) *(2026)* — Decouples generation from training; sparse updates plus Δ-IFD filtering; 8.1× speedup over on-policy baselines.
- [Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning](https://arxiv.org/abs/2605.07804) *(2026)* — Top-k overlap monitors prefix drift; attenuates unreliable rewards and truncates drifted rollouts.
- [EffOPD: Learning to Foresee — Unlocking Efficiency of On-Policy Distillation](https://arxiv.org/abs/2605.11739) *(2026)* — Adaptively extrapolates along the current update step for ~3× training acceleration with no extra trainables.

## Taxonomy

Cross-cutting views over the canonical papers. Many entries span multiple categories — this is for orientation, not strict partitioning.

### By Teacher Type

| Teacher Type | Papers |
|---|---|
| External white-box | MiniLLM, GKD, DistiLLM, DistiLLM-2, Veto, Entropy-Aware OPD, ExOPD, REOPOLD, PACED, Prefix OPD, Revisiting OPD, Rethinking OPD, Lightning OPD, Uni-OPD, SOD, AOPD, vOPD, SCOPE, HPD, TIP, DP-OPD, NPD, Prune-OPD, EffOPD, CoDistill-GRPO, Rock Tokens, Sparse-to-Dense |
| External black-box | Black-Box OPD / GAD, OVD, ROPD |
| Self-teacher with privileged context | OPSD, SDFT, SDPO, OPSDC, GATES, pi-Distill, RLSD, SDZero, OGLS-SD, PBSD, UniSD, ATESD, RLRT, EGRSD, CREDIT, SDAR, MixSD, AntiSD, TRACE, AVSD, VPD, RMSD, SPD, MSD-Safety, COPSD |
| Context-conditioned | OPCD, OEL |
| Multiple / lifecycle teachers | MiMo-V2-Flash MOPD, GLM-5, Qwen3, Baichuan-M3, DeepSeek-V4, CoPD, MAD-OPD, KAT-Coder-V2 |

### By Primary Goal

| Goal | Papers |
|---|---|
| Compression / strong-to-weak transfer | MiniLLM, GKD, Qwen3, Prefix OPD, Rethinking OPD, Lightning OPD |
| Post-RL consolidation / skill integration | MiMo MOPD, GLM-5, ExOPD, CoPD |
| Continual learning | SDFT, OPCD, OEL, MixSD |
| RL replacement / augmentation | SDPO, RLTF-SD, RLAD, REOPOLD, RLSD, SDZero, OGLS-SD, PBSD, CoDistill-GRPO, RLRT, EGRSD, CREDIT, SDAR, Sparse-to-Dense, AntiSD, TRACE, AVSD, VPD, RMSD |
| Reasoning compression | OPSDC |
| Black-box distillation | GAD, OVD, ROPD |

## Adjacent and Enabling Work

Papers that are not canonical OPD but matter for understanding or deploying it.

### Cross-Tokenizer and Model-Family Enablers

- [ULD: Towards Cross-Tokenizer Distillation](https://arxiv.org/abs/2402.12030) *(2024)* — Universal Logit Distillation; foundational enabler for cross-family OPD.
- [Multi-Level OT for Universal Cross-Tokenizer KD](https://arxiv.org/abs/2412.14528) *(2024)* — Token- and sequence-level optimal transport for cross-tokenizer KD.
- [CDM: Enhancing Cross-Tokenizer KD with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104) *(2025)* — Contextual dynamic mapping for vocabulary alignment.
- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083) *(2025)* — Approximate likelihood matching across fundamentally different tokenizers.
- [Cross-Tokenizer Likelihood Scoring Algorithms](https://arxiv.org/abs/2512.14954) *(2025)* — Exact and approximate sequence likelihood scoring across BPE vocabularies.
- [DSKD: A Dual-Space Framework for General KD](https://arxiv.org/abs/2504.11426) *(2025)* — Unifies output spaces; supports on- and off-policy KD between any two LLMs.
- [GOLD: Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) *(2025)* — Cross-tokenizer OPD with TRL integration.
- [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865) *(2026)* — Aligned-span projection plus teacher-anchored DPO with cross-tokenizer importance sampling.
- [DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer KD](https://arxiv.org/abs/2602.21669) *(2026)* — Dual-space token weighting plus Soft-DTW differentiable sequence alignment.
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466) *(2026)* — Byte-level conversion of teacher distributions plus byte-level student decoder for mismatched tokenizers.
- [SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2605.07711) *(2026)* — Short multi-token continuations replace exact matching; recovers teacher signal at mismatched positions.

### Mismatch Mitigation and Student Quality

- [Exploring and Enhancing Distribution Transfer in KD](https://arxiv.org/abs/2409.12512) *(2024)* — Analyzes reverse-KL with student-generated output; proposes OKD.
- [FIRST: Efficient Trustworthy Distillation](https://arxiv.org/abs/2408.12168) *(2024)* — Teacher recalibration for trustworthy offline KD.
- [Multi-Granularity Semantic Revision](https://arxiv.org/abs/2407.10068) *(2024)* — Sequence correction for low-quality student-generated outputs.
- [Warmup-Distill](https://arxiv.org/abs/2502.11766) *(2025)* — Bridges distribution mismatch before distillation begins.
- [TAID: Temporally Adaptive Interpolated Distillation](https://arxiv.org/abs/2501.16937) *(2025)* — Addresses teacher-student mismatch via adaptive interpolation.
- [SpecKD: Speculative Decoding for Effective KD](https://arxiv.org/abs/2510.24021) *(2025)* — Speculative-decoding-inspired selective token-level losses.
- [Knowledge Distillation with Training Wheels](https://arxiv.org/abs/2502.17717) *(2025)* — Entropy-regularized value optimization with on-/off-policy demonstrations.
- [Revealing the Power of Post-Training via KD](https://arxiv.org/abs/2509.26497) *(2025)* — Offline on-policy KD: student generates, then teacher labels.
- [TSD-KD: Explain in Your Own Words](https://arxiv.org/abs/2603.13260) *(2026)* — Student proposes candidates, teacher reranks, selective token distillation.
- [SSD: Embarrassingly Simple Self-Distillation Improves Code Generation](https://arxiv.org/abs/2604.01193) *(2026)* — Temperature-shifted self-sampling plus SFT; identifies precision-exploration conflict.
- [AdaSwitch: Balancing Exploration and Guidance in KD via Adaptive Switching](https://arxiv.org/abs/2510.07842) *(2025)* — Switches between on-policy rollouts and off-policy teacher data via context-aware divergence threshold.
- [DDT: Towards On-Policy SFT via Distribution Discriminant Theory](https://arxiv.org/abs/2602.12222) *(2026)* — In-Distribution Finetuning and Hinted Decoding realign training data to the student's distribution.
- [DASD: Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning](https://arxiv.org/abs/2601.09088) *(2026)* — On-policy correction pipeline for distribution mismatch and exposure bias in sequence-level CoT distillation.
- [Distillation Traps and Guards: A Calibration Knob for LLM Distillability](https://arxiv.org/abs/2604.18963) *(2026)* — Post-hoc calibrates teachers via RFT to control distillability against tail noise and instability.

### Preference, Reward-Guided, and Hybrid RL+KD

- [Direct Preference Knowledge Distillation](https://arxiv.org/abs/2406.19774) *(2024)* — Preference-aware KD combining reverse-KL with implicit reward objectives.
- [Online Knowledge Distillation with Reward Guidance](https://arxiv.org/abs/2505.18952) *(2025)* — Sequential KD via preference optimization; offline and online variants.
- [KDRL](https://arxiv.org/abs/2506.02208) *(2025)* — Unified reverse-KL KD with RL in a single post-training objective.
- [RLTF-SD: Expanding RL via Text Feedback](https://arxiv.org/abs/2602.02482) *(2026)* — Internalizes text feedback via self-distillation.
- [RLAD: Reinforcement-aware KD for LLM Reasoning](https://arxiv.org/abs/2602.22495) *(2026)* — Trust-region ratio distillation on student rollouts.
- [Multi-Token Prediction via Self-Distillation](https://arxiv.org/abs/2602.06019) *(2026)* — Online self-distillation for multi-token prediction and faster inference.
- [ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](https://arxiv.org/abs/2509.25100) *(2025)* — Mixed-policy preference distillation with student-generated outputs; black-box cross-architecture transfer.
- [SRPO: Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing](https://arxiv.org/abs/2604.02288) *(2026)* — Routes correct student rollouts to reward-based RL and failed ones to self-distillation.
- [KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation](https://arxiv.org/abs/2504.19024) *(2025)* — K-step Bellman return replaces high-variance single-step REINFORCE in sequence-level OPD.
- [Rethinking LLM Distillation: A Constrained MDP Perspective](https://arxiv.org/abs/2509.22921) *(2025)* — Maximizes task reward under hard KL constraint against the teacher; avoids manual Lagrangian tuning.
- [RLKD: Distilling LLMs' Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.16142) *(2025)* — Generative Structure Reward Model on student rollouts; outperforms SFT-RL pipelines on 0.1% data.
- [LUFFY: Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945) *(2025)* — Mixed-policy GRPO combining on-policy rollouts with off-policy teacher traces via regularized importance sampling.
- [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/abs/2407.14622) *(2024)* — RL mimicking best-of-N via Jeffreys-divergence matching; eliminates inference-time BoN cost.
- [Faster WIND: Accelerating Iterative Best-of-N Distillation for LLM Alignment](https://arxiv.org/abs/2410.20727) *(2024)* — Game-theoretic iterative BoN as self-play; win-rate dominance optimization (AISTATS 2025).
- [AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](https://arxiv.org/abs/2503.02832) *(2025)* — Casts RLHF as token-level distillation by injecting DPO rewards (ACL 2025).
- [KEPO: Knowledge-Enhanced Preference Optimization for Reinforcement Learning with Reasoning](https://arxiv.org/abs/2602.00400) *(2026)* — Quality-gated OPD on high-quality trajectories plus knowledge-enhanced exploration via teacher hints.
- [𝒳-KD: General Experiential Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2602.12674) *(2026)* — Jointly models teacher reward and policy-distills so the student learns inside the teacher's original environment.
- [ExGRPO: Probing to Refine — Reinforcement Distillation of LLMs via Explanatory Inversion](https://arxiv.org/abs/2603.19266) *(2026)* — Explanatory probes plus dialogue-structure utility bonus reward coherent reasoning over memorized answers.
- [NPO: Near-Future Policy Optimization](https://arxiv.org/abs/2604.20733) *(2026)* — Later checkpoint of same policy as teacher; AutoNPO adaptively triggers switch to maximize RLVR signal.
- [CoDistill-GRPO: A Co-Distillation Recipe for Efficient Group Relative Policy Optimization](https://arxiv.org/abs/2605.08873) *(2026)* — Dual GRPO with on-policy KD reward between large/small models; matches standard GRPO with 18% speedup.

### Self-Play and Iterative Bootstrapping

- [SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) *(2024)* — Self-play distinguishing own generations from human references (ICML 2024).
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) *(2024)* — Iterative DPO with model-as-judge self-rewards on own generations.
- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519) *(2025)* — MCTS-guided self-evolution; policy and PRM co-improve via code-augmented reasoning.
- [rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722) *(2025)* — GRPO with Resample-on-Correct rollouts plus multi-stage SFT→RL recipe for 14B agentic reasoner.
- [π-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data](https://arxiv.org/abs/2604.14054) *(2026)* — Examiner-generated tasks plus question-construction-paths as privileged context for dense student supervision.
- [SPHERE: Self-Evolved Preference Optimization for Mathematical Reasoning in SLMs](https://arxiv.org/abs/2503.04813) *(2025)* — PRM/ORM-scored MCTS rollouts plus self-correction yield preference pairs for iterative DPO.
- [SGS: Scaling Self-Play with Self-Guidance](https://arxiv.org/abs/2604.20209) *(2026)* — Three-role self-play (Solver, Generator, Reviewer) for theorem proving; 7B beats 671B at pass@4 on Lean4.

### Precursors

- [Autoregressive KD through Imitation Learning](https://arxiv.org/abs/2009.07253) *(2020)* — Early precursor framing sequence-model KD as imitation learning.
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) *(2022)* — Context distillation; key precursor to OPCD and OEL.

## Domain Extensions

OPD applied to non-text-reasoning settings — agents, multimodal models, diffusion, audio, robotics — and to inference acceleration via speculative decoding. These pass the inclusion criterion (student rollouts central to the learning signal) but on substrates beyond LLM text reasoning.

### Agent, Multimodal, and Other Extensions

- [Structured Agent Distillation](https://arxiv.org/abs/2505.13820) *(2025)* — Queries teacher online to avoid distribution drift in agent settings.
- [From Deferral to Learning: Online In-Context KD for LLM Cascades](https://arxiv.org/abs/2509.22984) *(2025)* — Teacher-student cascade with reusable online knowledge store.
- [AllMem](https://arxiv.org/abs/2602.13680) *(2026)* — Offline on-policy distillation for long-context modeling.
- [Video-OPD](https://arxiv.org/abs/2602.02994) *(2026)* — OPD for temporal video grounding in multimodal LLMs.
- [Reinforced Attention Learning](https://arxiv.org/abs/2602.04884) *(2026)* — On-policy attention distillation for multimodal models.
- [SCoRe: From Correction to Mastery via Reinforced Distillation of LLM Agents](https://arxiv.org/abs/2509.14257) *(2025)* — Teacher intervenes at first critical error in student agent trajectories for corrective distillation.
- [VOLD: Reasoning Transfer from LLMs to Vision-Language Models via OPD](https://arxiv.org/abs/2510.23497) *(2025)* — Text-only teacher distills reasoning into VLM via student-generated traces with combined GRPO and OPD.
- [X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs](https://arxiv.org/abs/2603.24596) *(2026)* — Student on-policy rollouts with token-level teacher feedback for cross-modal speech-LLM distillation.
- [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via OPD](https://arxiv.org/abs/2603.26666) *(2026)* — Reverse-KL OPD bridging offline SFT and online RL for robotic manipulation.
- [TCOD: Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents](https://arxiv.org/abs/2604.24005) *(2026)* — Short-to-long trajectory-depth curriculum mitigating multi-turn KL instability.
- [LLM4Teach: Large Language Model as a Policy Teacher for Training RL Agents](https://arxiv.org/abs/2311.13373) *(2023)* — LLM teacher distills into small RL agent that surpasses teacher through environment interaction.
- [RPD: Refined Policy Distillation — From VLA Generalists to RL Experts](https://arxiv.org/abs/2503.05833) *(2025)* — Teacher VLA actions guide student during RL exploration; combines RL with behavioral cloning (IROS 2026).
- [π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974) *(2025)* — Imitation distillation aligns student flow-model trajectories with teacher under standard flow matching (ICLR 2026).
- [Step-Audio-R1 Technical Report](https://arxiv.org/abs/2511.15848) *(2025)* — Modality-Grounded Reasoning Distillation produces audio reasoning grounded in acoustic features.
- [OPD-AVMP: On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning](https://arxiv.org/abs/2604.07944) *(2026)* — Generalized OPD for LLM-based driving planners; 5× compression at near-teacher performance.
- [CORD: Bridging the Audio–Text Reasoning Gap via Weighted On-policy Cross-modal Distillation](https://arxiv.org/abs/2601.16547) *(2026)* — Audio-conditioned rollouts; text-conditioned same model as teacher; importance-weighted reverse KL plus GRPO.
- [Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents](https://arxiv.org/abs/2604.10674) *(2026)* — Plain-prompt student; skill-augmented same model as token-level self-teacher for multi-turn agent training.
- [CoPD: Co-Evolving Policy Distillation](https://arxiv.org/abs/2604.27083) *(2026)* — Parallel expert training with bidirectional OPD; experts co-evolve as mutual teachers during RLVR.
- [PRISM: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL](https://arxiv.org/abs/2604.28123) *(2026)* — Black-box OPD pre-alignment between SFT and RLVR for VLMs; MoE discriminator supplies adversarial signals.
- [D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models](https://arxiv.org/abs/2605.05204) *(2026)* — OPSD ported to few-step T2I diffusion; text-only student vs. text+image teacher with velocity-MSE on rollouts.
- [Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063) *(2026)* — Per-domain Flow-GRPO experts supervise student SDE rollouts via reverse-KL with Manifold Anchor Regularization.
- [VISD: Enhancing Video Reasoning via Structured Self-Distillation](https://arxiv.org/abs/2605.06094) *(2026)* — Structured video judge feeds EMA teacher with privileged feedback; direction-magnitude decoupling stabilizes RL+supervision.
- [TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate Diffusion LLM](https://arxiv.org/abs/2605.09536) *(2026)* — Partitions masked positions by remaining decoding steps into near (CE) and distant (KL) subsets. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/BHmingyang/TAD)
- [DiMO: Distilling Masked Diffusion Models into One-step Generator](https://arxiv.org/abs/2503.15457) *(2025)* — First OPD for masked discrete diffusion image generation; Generalized Jeffrey divergence with DMD-style auxiliary (ICCV 2025).
- [SDAR: Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) *(2026)* — Sigmoid-gated OPSD auxiliary on top of GRPO for multi-turn agents; amplifies positive-gap, attenuates negative-gap tokens (COLM 2026). [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/ZJU-REAL/SDAR)
- [GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation](https://arxiv.org/abs/2605.11853) *(2026)* — Reverse-KL between student and ground-truth-conditioned same-model teacher gives token saliency; KL-initiated entropy-terminated segmentation propagates credit within segments to sign-aware reweight GRPO advantages.
- [Revisiting DAgger in the Era of LLM-Agents](https://arxiv.org/abs/2605.12913) *(2026)* — Turn-level (DAgger) and trajectory-prefix (AggreVaTe) student/teacher rollout mixtures with teacher actions queried at every visited state.
- [DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2605.15055) *(2026)* — Lifts OPD from autoregressive tokens to diffusion denoising via closed-form reverse-KL along student rollouts; unifies SDE and ODE samplers.
- [Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning](https://arxiv.org/abs/2605.22511) *(2026)* — Alternates GRPO with offline self-distillation on correct-fewest-search / divergent-sibling pairs mined from the converged rollout pool.
- [Healthcare AI GYM for Medical Agents](https://arxiv.org/abs/2605.02943) *(2026)* — Clinical-agent gymnasium plus Turn-level Truncated OPD: an EMA teacher conditioned on outcome-privileged hints that are stripped before logprob comparison.
- [HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents](https://arxiv.org/abs/2605.07177) *(2026)* — Parallel multimodal search agent that applies OPD only to failed rollouts to salvage correct intermediate tool calls from GRPO's uniform negative advantage. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](https://github.com/DeepExperience/HyperEyes)
- [DGPO: Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](https://arxiv.org/abs/2508.20324) *(2025)* — Selective reverse-KL teacher guidance on student-generated outputs within PPO; teacher intervenes only when the compact agent's autonomous attempts fail.
- [LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation](https://arxiv.org/abs/2512.23576) *(2025)* — On-policy distillation for real-time interactive video diffusion, extending the Self-Forcing few-step student-rollout recipe.
- [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://arxiv.org/abs/2605.13724) *(2026)* — On-policy flow-map distillation along the student's own Euler rollout; supports any-step video generation.

### Speculative Decoding (Draft-Model Training)

Draft-model training for speculative decoding shares OPD's core loop: the draft (student) generates, the target (teacher) verifies, and the draft is updated to match. Included for breadth even though the goal is inference acceleration rather than student capability.

- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) *(2023)* — Continuously updates draft on observed queries via KD; 1.42×-2.17× latency gains.
- [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) *(2023)* — Aligns draft with target via on-policy data and task-tailored divergence (ICLR 2024).
- [HASS: Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766) *(2024)* — Harmonized objective and context distillation fixes train-decoding inconsistency.
- [Falcon: Faster and Parallel Inference through Enhanced Semi-Autoregressive Drafting](https://arxiv.org/abs/2412.12639) *(2024)* — Coupled Sequential Glancing Distillation strengthens inter-token dependencies in semi-AR drafters.
- [CORAL: Consistent Representations across Multi-step Training with Lighter Speculative Drafter](https://arxiv.org/abs/2502.16880) *(2025)* — Cross-step representation alignment for multi-step drafter training (ACL 2025).
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) *(2025)* — Direct token prediction with multi-layer feature fusion under on-policy training-time test; up to 6.5×.
- [MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of VLMs](https://arxiv.org/abs/2505.10526) *(2025)* — Adapts SLM into VLM drafter via self-distilled visual instruction tuning.
- [DVI: Draft, Verify, and Improve — Toward Training-Aware Speculative Decoding](https://arxiv.org/abs/2510.05421) *(2025)* — Self-speculative drafter trained online from verifier decisions via KL→RL schedule.
- [ReSpec: Optimizing Speculative Decoding in Reinforcement Learning Systems](https://arxiv.org/abs/2510.26475) *(2025)* — Evolves drafter during RL via reward-weighted distillation on rollouts.
- [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting](https://openreview.net/forum?id=CRgWv0kWjF) *(2026)* — Multimodal speculative-reasoning drafter with verifier-gated parallel execution.
- [MSD: Speculative Decoding Reimagined for Multimodal Large Language Models](https://arxiv.org/abs/2505.14260) *(2025)* — Decouples text/visual tokens in draft; two-stage training lifts MLLM speedups to 2.29–2.46×.
- [SpecVLM: Fast Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2509.11815) *(2025)* — Elastic visual compressor plus online-logit distillation; 2.5–2.9× end-to-end VLM speedups.
- [ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](https://arxiv.org/abs/2509.15235) *(2025)* — Lightweight vision adaptor compresses image tokens; trained on target-generated long responses.
- [Aurora: When RL Meets Adaptive Speculative Training](https://arxiv.org/abs/2602.06932) *(2026)* — Online continual draft training; target verifications stream into FKL/RKL fine-tuning then hot-swap into serving.
- [SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting](https://arxiv.org/abs/2605.07243) *(2026)* — Block-iterative drafter with layer-wise shift; valid-prefix masking and cost-aware bandit adaptation.
- [SFDD: Flatter Tokens are More Valuable for Speculative Draft Model Training](https://arxiv.org/abs/2601.18902) *(2026)* — Sample-level flatness filters EAGLE training data; 2× speedup at 50% data with <4% inference-speedup loss.
- [OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding](https://arxiv.org/abs/2507.02659) *(2025)* — Online on-policy distillation on the draft's own generated tokens; cross-vocabulary n-gram cache lets one drafter serve any target.

## Technical Reports and Industrial Recipes

Production training pipelines that use OPD as a post-training stage.

| Year | System | OPD Usage | Link |
|------|--------|-----------|------|
| 2024 | Gemma 2 | KD as alternative to next-token prediction for 2B and 9B students | [arXiv](https://arxiv.org/abs/2408.00118) |
| 2025 | Qwen3 | Strong-to-weak; off-policy then on-policy distillation | [arXiv](https://arxiv.org/abs/2505.09388) |
| 2025 | Qwen3-Omni | Off-policy then on-policy distillation before GSPO | [arXiv](https://arxiv.org/abs/2509.17765) |
| 2025 | GLM-4.5 / 4.6 | Multi-stage post-training with expert model iteration and RL | [arXiv](https://arxiv.org/abs/2508.06471) |
| 2025 | HY-MT1.5 | Multi-stage translation: SFT + OPD + RL | [arXiv](https://arxiv.org/abs/2512.24092) |
| 2026 | MiMo-V2-Flash | Multi-Teacher OPD (MOPD) as post-training stage | [arXiv](https://arxiv.org/abs/2601.02780) |
| 2026 | GLM-5 | On-policy cross-stage distillation to recover earlier skills | [arXiv](https://arxiv.org/abs/2602.15763) |
| 2026 | Typhoon-S | Minimal sovereign recipe: SFT + OPD + small-scale RFT | [arXiv](https://arxiv.org/abs/2601.18129) |
| 2026 | Nemotron-Cascade 2 | Cascade RL + multi-domain on-policy distillation | [arXiv](https://arxiv.org/abs/2603.19220) |
| 2026 | Baichuan-M3 | Task RL → offline policy distillation → multi-teacher OPD | [arXiv](https://arxiv.org/abs/2602.06570) |
| 2026 | MobileLLM-R1.5 | Final-stage on-policy KD as primary improvement over R1 | [model card](https://huggingface.co/facebook/MobileLLM-R1.5-950M) |
| 2026 | Nanbeige4-3B-Thinking | OPD preferred over off-policy for math reasoning | [model card](https://huggingface.co/Nanbeige/Nanbeige4-3B-Thinking-2510) |
| 2026 | DeepSeek-V4 | Domain-expert SFT+GRPO → unified model consolidation via OPD | [report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) |
| 2026 | Qwen3.5-Omni | Specialist distillation → privileged-input self-distillation aligning audio to text | [arXiv](https://arxiv.org/abs/2604.15804) |
| 2026 | HY-Embodied-0.5 | 32B → 2B on-policy distillation; student rollouts, teacher token-level supervision | [arXiv](https://arxiv.org/abs/2604.07430) |
| 2026 | KAT-Coder-V2 | Specialize-then-Unify: 5 domain-expert agents → unified via OPD on student trajectories | [arXiv](https://arxiv.org/abs/2603.27703) |
| 2026 | Cursor Composer 2.5 | Hint-conditioned self-teacher OPD KL added to RL for targeted behaviors (tool calls, style); built on Kimi K2.5 | [blog](https://cursor.com/blog/composer-2-5) |

## Frameworks and Implementations

### Training Frameworks

| Framework | Description | Link |
|---|---|---|
| TRL | GKD, GOLD, and MiniLLM trainers; most accessible starting point | [docs](https://huggingface.co/docs/trl) |
| NeMo-RL | Multi-teacher and cross-tokenizer OPD at scale | [docs](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html), [repo](https://github.com/NVIDIA-NeMo/RL) |
| veRL | Async on-policy KD trading strict on-policy guarantees for throughput | [docs](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) |
| MS-Swift | GKD and OPSD sections in the ModelScope ecosystem | [docs](https://swift.readthedocs.io/en/latest/) |
| EasyDistill | Comprehensive KD toolkit for black-box and white-box LLM distillation | [arXiv](https://arxiv.org/abs/2505.20888) |
| KDFlow | Off-policy, on-policy, and cross-tokenizer distillation via decoupled backends | [arXiv](https://arxiv.org/abs/2603.01875), [repo](https://github.com/songmzhang/KDFlow) |
| slime | Unified RL stack supporting on-policy distillation and hindsight hints | [repo](https://github.com/THUDM/slime) |
| OpenClaw-RL | Agentic RL stack with hindsight-guided OPD | [arXiv](https://arxiv.org/abs/2603.10165) |
| NexRL | Dedicated on-policy distillation recipes | [repo](https://github.com/nex-agi/NexRL) |
| SkyRL | OPD examples and blog resources | [repo](https://github.com/NovaSky-AI/SkyRL) |
| ATLAS | Continual-learning framework using GKD/GRPO from runtime traces | [docs](https://docs.arc.computer/introduction) |
| AReaL | OPD and KDRL over student-sampled trajectories with teacher log-prob guidance | [docs](https://github.com/inclusionAI/AReaL/blob/main/docs/en/algorithms/distillation.md) |
| rLLM | Agent RL framework (UC Berkeley Sky) with first-class OPD: `examples/math_distill/` (DeepMath OPSD + `train_deepmath_distill_tinker.{py,sh}`) and `rllm/trainer/distill/` modules over verl or tinker backends | [docs](https://docs.rllm-project.com/), [repo](https://github.com/rllm-org/rllm) |
| SpecForge | Speculative draft training with EAGLE-3 support and hybrid parallelism | [arXiv](https://arxiv.org/abs/2603.18567), [repo](https://github.com/sgl-project/SpecForge) |
| TorchSpec | Torch-native speculative draft training with disaggregated inference/training; streams target hidden states via Mooncake store; Kimi-K2.5/MiniMax-M2.5/Qwen3-Coder-Next examples | [blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/), [repo](https://github.com/lightseekorg/TorchSpec) |
| Tinker Cookbook | Thinking Machines' Tinker SDK recipes for off-policy KD, single/multi-teacher OPD, multi-turn tool use | [recipes](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation), [repo](https://github.com/thinking-machines-lab/tinker-cookbook) |
| ROLL | Alibaba's scalable RL library for LLMs/VLMs with an OPD pipeline | [repo](https://github.com/alibaba/ROLL) |

### Implementations

- [OPSD](https://github.com/siyan-zhao/OPSD) — Official code for Self-Distilled Reasoner / OPSD.
- [SCOPE](https://github.com/machine981/SCOPE) — Dual-path OPD: student-PPL-weighted MLE for correct rollouts, teacher-PPL-weighted KL for incorrect.
- [CaOPD](https://github.com/SalesforceAIResearch/CaOPD) — K student rollouts → empirical success rate → confidence target replacement → reverse-KL OPD.
- [OPSD-OnPolicyDistillation](https://github.com/HJSang/OPSD_OnPolicyDistillation) — verl-based OPD with separate teacher, agent-loop rollouts, and memory-efficient execution.
- [nano-opd](https://github.com/Athe-kunal/nano-opd) — Hackable OPD library decoupling vLLM rollout, FSDP training, and teacher forwards across independent GPU groups.

## Acknowledgments

This list draws on the parallel curation effort at [thinkwee/AwesomeOPD](https://github.com/thinkwee/AwesomeOPD), which provided pointers to several papers (notably speculative-decoding draft training, BoN distillation, self-play, multilingual and crosslingual self-distillation, clinical and multimodal agentic OPD, additional industrial reports, and several training frameworks). The two lists organize differently — thinkwee/AwesomeOPD groups by feedback signal and access mode; this list groups by methodological role — and are best read together.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for criteria, section placement, and formatting.

- **Inclusion criteria:** the work should involve student rollouts as central to the learning signal, or directly enable OPD deployment (cross-tokenizer, frameworks, etc.).
- **Entry format:** `[Title](url) *(Year)* — One-line description.` See [CONTRIBUTING.md](CONTRIBUTING.md) for full examples.

## Citation

```bibtex
@software{awesome-on-policy-distillation,
  title = {{Awesome On-Policy Distillation}},
  author = {Liu, Chris Yuhao and others},
  year = {2026},
  doi = {10.5281/zenodo.19411493},
  url = {https://github.com/chrisliu298/awesome-on-policy-distillation},
  version = {v1.0.0}
}
```

---
