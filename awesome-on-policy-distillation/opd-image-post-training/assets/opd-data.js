window.OPD_IMAGE_DATA = {
  capturedAt: "2026-05-26",
  method: {
    name: "On-Policy Distillation",
    shortName: "OPD",
    description:
      "学生模型从自己的当前策略采样，并在这些 on-policy 状态上接收教师信号。它把 distillation 的稠密监督和 online RL 的状态局部性结合起来。",
    sourceTitle: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
    sourceUrl: "https://paperswithcode.co/paper/2306.13649",
    introducedYear: 2023,
    paperCount: 224,
    pwcUrl: "https://paperswithcode.co/methods/on-policy-distillation",
    apiUrl: "https://paperswithcode.co/api/v1/methods/on-policy-distillation",
    trending: ["Flow-OPD", "AnyFlow", "D-OPSD", "Revisiting OPD"]
  },
  pages: [
    {
      slug: "trajectory-opd",
      file: "trajectory-opd.html",
      category: "mechanism",
      eyebrow: "机制基础",
      title: "从 Token 到图像轨迹",
      subtitle: "把 LLM 的 prefix rollout 翻译成 diffusion / flow 的 latent trajectory。",
      goal:
        "解释 OPD 的状态分布思想如何从离散 token 推广到连续图像生成过程，并建立 KL、velocity、mean matching 之间的对应关系。",
      coreQuestions: [
        "学生自己走过的状态在图像模型里是什么：mask token、latent x_t、ODE 子步，还是 SDE 轨迹？",
        "教师信号从 token logits 变成什么：velocity field、score、mean、flow map，还是 reward-shaped target？",
        "什么时候仍然是 distillation，什么时候已经更接近 dense RL？"
      ],
      concepts: [
        "Student rollout",
        "Trajectory-level KL",
        "Velocity matching",
        "SDE / ODE mean matching",
        "Train-test state alignment"
      ],
      papers: ["gkd", "minillm", "survey", "dimo", "piflow", "diffusionopd"],
      visuals: [
        "左栏展示 token prefix 逐步增长，右栏展示 latent 从噪声走向图像。",
        "公式卡：token KL -> trajectory KL -> velocity / mean matching。",
        "时间滑块：同一个 t 上比较 student state 与 teacher target。"
      ],
      recipe: [
        "定义部署时学生实际会使用的 sampler 和步数。",
        "用学生当前参数采样 latent trajectory，而不是重放教师轨迹。",
        "在学生访问的每个关键 timestep 查询教师 velocity、score 或 mean。",
        "用 trajectory matching loss 更新学生，并保留少量 anchor loss 防止漂移。"
      ],
      diagnostics: [
        ["状态错位", "学生轨迹落在教师未覆盖区域", "提高 rollout 频率，加入 anchor 或 top-k support 约束"],
        ["高方差", "少量 timestep 上 teacher signal 过尖", "按 entropy / timestep 重加权"],
        ["语义退化", "latent trajectory 对 prompt 条件不敏感", "加入 text-image alignment reward 或 teacher context"]
      ],
      acceptance: [
        "能用一句话说明图像 OPD 是在学生自己的生成轨迹上蒸馏教师。",
        "能区分 off-policy KD、sparse reward RL、on-policy trajectory matching。",
        "能指出至少两种图像模型里的 teacher signal 形态。"
      ]
    },
    {
      slug: "few-step-distillation",
      file: "few-step-distillation.html",
      category: "few-step",
      eyebrow: "少步生成",
      title: "少步与一步生成",
      subtitle: "用 OPD 缓解普通蒸馏在 few-step / one-step 图像生成里的轨迹错位。",
      goal:
        "说明为什么 few-step 图像模型继续训练时容易丢质量或多样性，以及 OPD 如何让学生在自己的短轨迹上被纠偏。",
      coreQuestions: [
        "普通蒸馏为什么会在 one-step / few-step 学生上产生质量和多样性的 tradeoff？",
        "学生 policy substep 采样如何减少 teacher-student trajectory mismatch？",
        "D-OPSD 式 privileged teacher 如何在微调时保护原有少步能力？"
      ],
      concepts: [
        "NFE",
        "Step-distilled diffusion",
        "Masked diffusion",
        "Policy-based flow",
        "Diversity collapse"
      ],
      papers: ["dimo", "piflow", "dopsd", "dmd", "dmd2"],
      visuals: [
        "NFE vs quality / diversity 曲线，用颜色区分 off-policy KD、OPD、RL。",
        "学生 policy substep 轨迹图：teacher 只在学生真实到达点上给速度。",
        "masked diffusion 一步补全示意。"
      ],
      recipe: [
        "从已有 step-distilled T2I 或 flow 模型出发，固定部署步数。",
        "采集学生少步 rollout，记录每步 latent、condition、scheduler state。",
        "用高步教师、EMA 教师或 privileged-context teacher 给同一状态打 velocity / distribution target。",
        "混合 OPD loss、少量原始 distillation loss 和图像质量偏好信号。",
        "用 NFE 固定的指标集评估，不允许通过增加步数掩盖退化。"
      ],
      diagnostics: [
        ["少步能力退化", "微调目标只优化最终图，忽视短轨迹", "固定 NFE 评估并加入 trajectory anchor"],
        ["多样性下降", "teacher target 过窄或 reverse KL 过强", "提高 entropy-aware 权重或混合 forward KL"],
        ["纹理过拟合", "小数据定制把学生推离基础分布", "保留基础 prompt replay 和 teacher consistency"]
      ],
      acceptance: [
        "页面明确说明 OPD 适合加速采样、保留多样性、继续微调少步模型。",
        "列出过拟合、少步能力退化、轨迹偏离教师三类失败信号。",
        "给出 DiMO、pi-Flow、D-OPSD 三条互补路线。"
      ]
    },
    {
      slug: "post-training-recipe",
      file: "post-training-recipe.html",
      category: "post-training",
      eyebrow: "工程配方",
      title: "图像生成后训练配方",
      subtitle: "把 OPD 放进真实 post-training pipeline：数据、teacher、rollout、loss、eval。",
      goal:
        "给出在图像生成模型后训练阶段使用 OPD 的可执行流程，并区分不需要 reward model 与需要 task-specific teacher 的两类分支。",
      coreQuestions: [
        "什么时候用 OPD 替代或补充 SFT、DreamBooth、DPO、GRPO？",
        "teacher 应该是大模型、EMA、任务专家、还是同模型的 privileged view？",
        "rollout、teacher forward 和 loss mixing 如何安排才不让成本失控？"
      ],
      concepts: [
        "Teacher access",
        "Privileged context",
        "Rollout cache",
        "Loss mixing",
        "Stop criteria"
      ],
      papers: ["dopsd", "flowopd", "diffusionopd", "exopd", "flowgrpo"],
      visuals: [
        "训练流程泳道图：prompt/data -> student rollout -> teacher guidance -> trajectory loss -> eval。",
        "配方选择器：few-step customization / multi-reward alignment / multi-task consolidation。",
        "成本面板：student rollout、teacher inference、reward model、eval 四类开销。"
      ],
      recipe: [
        "定义部署约束：模型族、步数、prompt 分布、是否允许 teacher logits / hidden states。",
        "选择 teacher：大步教师用于加速，任务专家用于多目标，privileged teacher 用于定制微调。",
        "按当前学生策略采样 rollout，并保存 timestep、condition、latent、noise seed、scheduler state。",
        "在同一学生状态上查询 teacher target，优先选择可稳定计算的 velocity / mean / distribution target。",
        "混合 OPD loss、anchor loss、少量 reward 或 preference loss；每轮更新后刷新 rollout。",
        "用固定 NFE、固定 prompt set、人工/自动指标和回归集决定停止。"
      ],
      diagnostics: [
        ["rollout 陈旧", "缓存样本来自旧学生", "设置刷新周期，跟踪 KL drift"],
        ["teacher 成本过高", "每步都查大教师", "只查关键 timesteps 或蒸馏中间教师"],
        ["目标冲突", "多个 reward 同时拉扯", "先训专家，再用 OPD consolidation"]
      ],
      acceptance: [
        "能输出一个完整 pipeline：数据、teacher、rollout、loss、评估指标、停止条件。",
        "明确无 reward model 分支和 task-specific teacher 分支。",
        "说明 rollout 刷新、teacher 查询和 anchor loss 的工程角色。"
      ]
    },
    {
      slug: "multi-task-alignment",
      file: "multi-task-alignment.html",
      category: "multi-task",
      eyebrow: "多任务整合",
      title: "多任务对齐与多教师整合",
      subtitle: "用 OPD 将单任务专家的探索能力蒸进统一图像生成模型。",
      goal:
        "研究 OPD 如何缓解多目标图像 RL 的 seesaw effect、reward hacking 和遗忘，并借鉴工业 MOPD 的能力整合思路。",
      coreQuestions: [
        "多 reward 联合训练为什么常常互相干扰？",
        "先训练专家再 OPD 到统一学生是否比 joint RL 更稳定？",
        "任务路由、专家冲突和 replay anchor 应如何设计？"
      ],
      concepts: [
        "Multi-teacher OPD",
        "Task routing",
        "Expert consolidation",
        "Seesaw effect",
        "Manifold anchor"
      ],
      papers: ["flowopd", "diffusionopd", "exopd", "rethinking", "qwen3"],
      visuals: [
        "任务专家到统一学生的 Sankey 图。",
        "多指标雷达图：OCR、GenEval、aesthetic、human preference。",
        "joint RL / cascade RL / OPD consolidation 三路对比。"
      ],
      recipe: [
        "按目标拆分训练集和 reward：文本渲染、物体关系、美学、人类偏好等。",
        "为每个目标训练或选择一个 teacher / expert，避免早期多目标互相抵消。",
        "统一学生按当前策略采样，并用 task routing 选择对应专家信号。",
        "对每类任务设置不同 loss 权重、anchor 强度和刷新频率。",
        "定期在全任务回归集上检查短板，必要时提高对应专家采样率。"
      ],
      diagnostics: [
        ["指标跷跷板", "单一更新同时优化冲突 reward", "拆成专家探索 + OPD 整合"],
        ["专家冲突", "同一 prompt 被不同 teacher 拉向不同风格", "加入 task label 和 gating"],
        ["遗忘", "新任务覆盖基础能力", "基础 prompt replay 与 manifold anchor"]
      ],
      acceptance: [
        "能解释单任务探索和多任务整合为什么要解耦。",
        "至少比较 joint RL、cascade RL、OPD consolidation 三条路径。",
        "包含 OCR、GenEval、aesthetic、human preference 的任务过滤或指标示意。"
      ]
    },
    {
      slug: "failure-diagnostics",
      file: "failure-diagnostics.html",
      category: "diagnostics",
      eyebrow: "稳定性诊断",
      title: "失败模式与诊断",
      subtitle: "把 LLM OPD 的稳定性研究转译到图像生成 post-training。",
      goal:
        "把 teacher-student mismatch、entropy collapse、trajectory drift 等 OPD 风险变成可观察指标和修复动作。",
      coreQuestions: [
        "teacher 在学生偏离轨迹上是否仍然可靠？",
        "怎样发现 entropy collapse、support mismatch、gradient variance？",
        "图像质量、prompt alignment 和少步能力如何同时监控？"
      ],
      concepts: [
        "Top-k overlap",
        "Entropy-aware loss",
        "Teacher-student compatibility",
        "Trajectory drift",
        "Anchor regularization"
      ],
      papers: ["revisiting", "entropyopd", "rethinking", "tip", "flowopd"],
      visuals: [
        "诊断仪表盘：divergence、entropy、reward、image quality 四条曲线。",
        "症状 -> 原因 -> 修复矩阵。",
        "timestep heatmap：定位哪些阶段 teacher signal 不稳定。"
      ],
      recipe: [
        "每轮记录学生与教师在关键 timestep 的 divergence、target norm 和 entropy。",
        "把评估分成最终图质量、prompt alignment、少步一致性、基础能力回归四组。",
        "对异常样本保留 seed、prompt、latent trajectory，方便复现。",
        "当 drift 或 collapse 超阈值时降低 OPD 权重，提升 anchor 或切换 teacher。"
      ],
      diagnostics: [
        ["teacher incompatibility", "强教师路径与学生能力重叠不足", "先做 teacher-student fit 评估，降低难度或换教师"],
        ["loss 高方差", "少数状态 teacher target 过强", "token/timestep importance weighting"],
        ["图像质量退化", "轨迹损失压过感知质量", "加入质量回归集和 image preference gate"],
        ["prompt alignment 退化", "visual fidelity 和 text condition 脱钩", "增加 caption / VLM reward 或条件一致性 loss"]
      ],
      acceptance: [
        "每个失败模式都有可观测指标和修复建议。",
        "覆盖 teacher incompatibility、loss 高方差、图像质量退化、prompt alignment 退化。",
        "诊断矩阵可以按症状切换查看。"
      ]
    }
  ],
  paperPool: {
    gkd: {
      title: "GKD: On-Policy Distillation of Language Models",
      year: 2023,
      link: "https://arxiv.org/abs/2306.13649",
      signal: "student rollout + flexible divergence",
      summary: "建立从 fixed KD 到 on-policy student-generated mistakes 的统一训练框架。",
      trust: "高可信"
    },
    minillm: {
      title: "MiniLLM: Knowledge Distillation of Large Language Models",
      year: 2023,
      link: "https://arxiv.org/abs/2306.08543",
      signal: "reverse KL on student generations",
      summary: "把生成式 LLM 蒸馏明确写成学生采样上的 reverse-KL 优化。",
      trust: "高可信"
    },
    survey: {
      title: "A Survey of On-Policy Distillation for Large Language Models",
      year: 2026,
      link: "https://arxiv.org/abs/2604.00626",
      signal: "taxonomy",
      summary: "按 teacher access、feedback signal、loss scope 和 training distribution 整理 OPD。",
      trust: "新综述"
    },
    dimo: {
      title: "DiMO: Distilling Masked Diffusion Models into One-step Generator",
      year: 2025,
      link: "https://arxiv.org/abs/2503.15457",
      signal: "token-level distribution matching + generalized Jeffrey divergence",
      summary: "将 masked discrete diffusion 图像模型压到 one-step generator，是图像 OPD 的强基线。",
      trust: "高可信"
    },
    piflow: {
      title: "pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation",
      year: 2025,
      link: "https://arxiv.org/abs/2510.14974",
      signal: "teacher velocity on student ODE trajectory",
      summary: "最直接展示图像 OPD 如何沿学生自己的 flow 轨迹匹配教师速度场。",
      trust: "高可信"
    },
    dopsd: {
      title: "D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models",
      year: 2026,
      link: "https://arxiv.org/abs/2605.05204",
      signal: "text-only student vs text+image privileged teacher",
      summary: "把 privileged context 自蒸馏用于 few-step T2I 持续微调，保护少步生成能力。",
      trust: "新论文/待复现"
    },
    flowopd: {
      title: "Flow-OPD: On-Policy Distillation for Flow Matching Models",
      year: 2026,
      link: "https://arxiv.org/abs/2605.08063",
      signal: "Flow-GRPO experts + reverse KL + manifold anchor",
      summary: "先训练单任务 flow experts，再沿统一学生 rollout 做多目标能力整合。",
      trust: "新论文/待复现"
    },
    diffusionopd: {
      title: "DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models",
      year: 2026,
      link: "https://arxiv.org/abs/2605.15055",
      signal: "continuous-state Markov OPD + per-step KL / mean matching",
      summary: "把 OPD 从自回归 token 推广到 diffusion denoising，并统一 SDE 与 ODE sampler。",
      trust: "新论文/待复现"
    },
    anyflow: {
      title: "AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation",
      year: 2026,
      link: "https://arxiv.org/abs/2605.13724",
      signal: "flow-map transition on student Euler rollout",
      summary: "将 on-policy flow-map distillation 用于任意步视频生成，是图像线的后续方向。",
      trust: "新论文/待复现"
    },
    livetalk: {
      title: "LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation",
      year: 2025,
      link: "https://arxiv.org/abs/2512.23576",
      signal: "improved self-forcing / on-policy video recipe",
      summary: "将 OPD 思路用于实时交互视频扩散，系统级指标需要继续核验。",
      trust: "待核验"
    },
    selfforcing: {
      title: "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion",
      year: 2025,
      link: "https://arxiv.org/abs/2506.08009",
      signal: "self-generated video rollout + holistic loss",
      summary: "不是标准 teacher-student OPD，但奠定视频生成中按测试轨迹训练的范式。",
      trust: "高可信"
    },
    dmd: {
      title: "Distribution Matching Distillation",
      year: 2023,
      link: "https://arxiv.org/abs/2311.18828",
      signal: "score / distribution matching baseline",
      summary: "one-step image distillation 的关键对照方法，用于理解 OPD 的轨迹优势。",
      trust: "高可信"
    },
    dmd2: {
      title: "DMD2: Improved Distribution Matching Distillation",
      year: 2024,
      link: "https://arxiv.org/abs/2405.14867",
      signal: "improved distribution matching",
      summary: "改进 DMD 的稳定性和质量，是 few-step OPD 需要比较的强基线。",
      trust: "高可信"
    },
    flowgrpo: {
      title: "Flow-GRPO: Training Flow Matching Models via Online RL",
      year: 2025,
      link: "https://arxiv.org/abs/2505.05470",
      signal: "reward + ODE-to-SDE exploration",
      summary: "不是蒸馏方法，但为 Flow-OPD 的专家预训练提供 post-training 前置技术。",
      trust: "高可信"
    },
    exopd: {
      title: "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation",
      year: 2026,
      link: "https://arxiv.org/abs/2602.12125",
      signal: "dense KL-constrained RL + reward extrapolation",
      summary: "把 OPD 明确解释为 dense RL，可以帮助图像 OPD 设计超越教师的 reward 分支。",
      trust: "新论文"
    },
    revisiting: {
      title: "Revisiting On-Policy Distillation",
      year: 2026,
      link: "https://arxiv.org/abs/2603.25562",
      signal: "failure modes + top-k support matching",
      summary: "诊断 OPD 的不稳定、信号失衡和 tokenizer mismatch，可转译为图像轨迹诊断。",
      trust: "新论文"
    },
    entropyopd: {
      title: "Entropy-Aware On-Policy Distillation",
      year: 2026,
      link: "https://arxiv.org/abs/2603.07079",
      signal: "entropy-aware forward KL on high-entropy states",
      summary: "提醒 OPD 不应盲目压低高熵状态，图像里对应多样性保护。",
      trust: "新论文"
    },
    rethinking: {
      title: "Rethinking On-Policy Distillation",
      year: 2026,
      link: "https://arxiv.org/abs/2604.13016",
      signal: "teacher-student compatibility",
      summary: "指出强教师不必然可教，关键在学生和教师的思考模式或轨迹重叠。",
      trust: "新论文"
    },
    tip: {
      title: "TIP: Token Importance in On-Policy Distillation",
      year: 2026,
      link: "https://arxiv.org/abs/2604.14084",
      signal: "importance weighting",
      summary: "将不同 token 的学习价值分开处理，可类比为图像 timestep / region importance。",
      trust: "新论文"
    },
    qwen3: {
      title: "Qwen3 Technical Report",
      year: 2025,
      link: "https://arxiv.org/abs/2505.09388",
      signal: "industrial off-policy + on-policy KD",
      summary: "工业 post-training 中 OPD 已成为强到弱蒸馏和能力迁移的常用阶段。",
      trust: "高可信"
    }
  },
  sources: [
    ["PapersWithCode OPD method page", "https://paperswithcode.co/methods/on-policy-distillation"],
    ["PapersWithCode OPD API snapshot", "https://paperswithcode.co/api/v1/methods/on-policy-distillation"],
    ["Local README", "../README.md"],
    ["Existing OPD visual guide", "../opd-visual-guide.html"],
    ["Existing OPD study guide", "../opd-study-guide.html"],
    ["Design reference", "../../claude/DESIGN.md"]
  ]
};
