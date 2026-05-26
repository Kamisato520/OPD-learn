(function () {
  const data = window.OPD_IMAGE_DATA;
  const categories = [
    ["all", "全部"],
    ["mechanism", "机制"],
    ["few-step", "少步生成"],
    ["post-training", "后训练"],
    ["multi-task", "多任务"],
    ["diagnostics", "诊断"]
  ];

  const recipes = [
    {
      id: "few",
      label: "few-step customization",
      title: "少步模型定制微调",
      steps: [
        "固定部署步数和 sampler，不通过增加 NFE 获得指标。",
        "学生按当前 few-step policy 采样 latent trajectory。",
        "teacher 使用 text+target image 或高步模型在学生状态上给 velocity target。",
        "混合 velocity-MSE、anchor replay 和少量偏好信号。",
        "用固定 prompt 回归集检查少步能力、风格一致性和多样性。"
      ]
    },
    {
      id: "reward",
      label: "multi-reward alignment",
      title: "多奖励对齐",
      steps: [
        "按 OCR、物体关系、美学、人类偏好拆分 reward 和数据。",
        "先单独训练任务专家或筛出高质量 teacher checkpoints。",
        "统一学生 rollout 后通过 task routing 选择 teacher guidance。",
        "用 OPD consolidation 替代直接 joint RL 的早期混训。",
        "用雷达图跟踪 seesaw effect，不只看平均分。"
      ]
    },
    {
      id: "consolidate",
      label: "multi-task consolidation",
      title: "多阶段能力整合",
      steps: [
        "把早期 checkpoint、任务专家和大步教师都视作可查询 teacher。",
        "对当前学生 rollout 采样，并记录 task label 和 scheduler state。",
        "在关键 timestep 查询对应 teacher，减少全步 teacher forward 成本。",
        "加入 manifold anchor 或基础 prompt replay 防止遗忘。",
        "按阶段能力回归集决定 teacher 权重和停止时机。"
      ]
    }
  ];

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function paper(id) {
    return data.paperPool[id];
  }

  function paperCard(id) {
    const item = paper(id);
    return `
      <article class="paper-card" data-trust="${escapeHtml(item.trust)}">
        <div class="paper-meta">
          <span>${item.year}</span>
          <span>${escapeHtml(item.trust)}</span>
        </div>
        <h3>${escapeHtml(item.title)}</h3>
        <dl>
          <div><dt>OPD 信号</dt><dd>${escapeHtml(item.signal)}</dd></div>
          <div><dt>摘要</dt><dd>${escapeHtml(item.summary)}</dd></div>
        </dl>
        <a class="text-link" href="${item.link}" target="_blank" rel="noopener noreferrer">阅读来源</a>
      </article>
    `;
  }

  function sourceList() {
    return data.sources
      .map(([label, href]) => `<li><a href="${href}" target="${href.startsWith("http") ? "_blank" : "_self"}" rel="noopener noreferrer">${escapeHtml(label)}</a></li>`)
      .join("");
  }

  function renderHome() {
    const app = $("#app");
    app.innerHTML = `
      <section class="hero home-hero">
        <div class="hero-copy">
          <div class="eyebrow"><span></span> OPD for image generation post-training</div>
          <h1>图像生成 OPD：让学生沿自己的去噪轨迹被教师纠偏。</h1>
          <p class="lead">这个静态研究站把 On-Policy Distillation 从 LLM token rollout 翻译到 diffusion、flow、few-step T2I 和多任务图像后训练。</p>
          <div class="hero-actions">
            <a class="button primary" href="#directions">进入研究方向</a>
            <a class="button secondary" href="post-training-recipe.html">查看后训练配方</a>
          </div>
        </div>
        <figure class="hero-media">
          <img src="../AwesomeOPD/banner.png" alt="OPD trajectory visual">
          <figcaption>Student trajectory, teacher guidance, dense correction.</figcaption>
        </figure>
      </section>

      <section class="section split-section" id="method">
        <div>
          <div class="eyebrow"><span></span> PapersWithCode snapshot</div>
          <h2>OPD 已被归为 post-training 方法。</h2>
          <p>${escapeHtml(data.method.description)}</p>
        </div>
        <aside class="fact-card">
          <div class="fact-row"><span>源论文</span><strong>${escapeHtml(data.method.sourceTitle)}</strong></div>
          <div class="fact-row"><span>Introduced</span><strong>${data.method.introducedYear}</strong></div>
          <div class="fact-row"><span>关联论文</span><strong>${data.method.paperCount}</strong></div>
          <div class="fact-row"><span>抓取日期</span><strong>${escapeHtml(data.capturedAt)}</strong></div>
          <a class="button secondary full" href="${data.method.pwcUrl}" target="_blank" rel="noopener noreferrer">打开 PWC 方法页</a>
        </aside>
      </section>

      <section class="section" id="directions">
        <div class="section-head">
          <div>
            <div class="eyebrow"><span></span> Research map</div>
            <h2>5 个子页面覆盖从机制到诊断的完整路线。</h2>
          </div>
          <p>筛选方向后进入对应页面。每个页面都包含核心问题、论文地图、机制图、后训练建议和验收 checklist。</p>
        </div>
        <div class="filter-row" aria-label="方向筛选">
          ${categories.map(([id, label]) => `<button class="filter-button ${id === "all" ? "active" : ""}" type="button" data-filter="${id}">${label}</button>`).join("")}
        </div>
        <div class="direction-grid">
          ${data.pages.map(page => `
            <article class="direction-card js-filter-card" data-kind="${page.category}">
              <span class="label">${escapeHtml(page.eyebrow)}</span>
              <h3>${escapeHtml(page.title)}</h3>
              <p>${escapeHtml(page.goal)}</p>
              <a class="button secondary" href="${page.file}">打开子页面</a>
            </article>
          `).join("")}
        </div>
      </section>

      <section class="dark-band" id="recipe">
        <div class="section">
          <div class="section-head">
            <div>
              <div class="eyebrow on-dark"><span></span> Post-training selector</div>
              <h2>后训练阶段先选问题，再选 teacher。</h2>
            </div>
            <p>OPD 的核心不是重放教师样本，而是让学生暴露自己真实会走到的状态，再在这些状态上获得稠密监督。</p>
          </div>
          <div class="recipe-layout">
            <div class="recipe-tabs" role="tablist" aria-label="后训练配方">
              ${recipes.map((item, index) => `<button class="recipe-tab ${index === 0 ? "active" : ""}" type="button" data-recipe="${item.id}">${item.label}</button>`).join("")}
            </div>
            <article class="recipe-panel" id="recipe-panel" aria-live="polite"></article>
          </div>
        </div>
      </section>

      <section class="section" id="papers">
        <div class="section-head">
          <div>
            <div class="eyebrow"><span></span> Paper map</div>
            <h2>关键论文池：基础 OPD、图像轨迹、少步生成、多任务整合。</h2>
          </div>
          <p>新近论文以研究线索呈现，成稿中保留“新论文/待复现”或“待核验”标签。</p>
        </div>
        <div class="paper-grid compact">
          ${["gkd", "dimo", "piflow", "dopsd", "flowopd", "diffusionopd", "anyflow", "revisiting"].map(paperCard).join("")}
        </div>
      </section>

      <section class="section source-section">
        <div class="callout">
          <h2>实施重点</h2>
          <p>图像 OPD 的最小可行版本：固定部署 sampler，让学生采样自己的 latent trajectory，在这些状态上查询 teacher velocity / mean / distribution target，并用 anchor loss 保护基础能力。</p>
        </div>
        <div class="source-list">
          <h3>来源</h3>
          <ol>${sourceList()}</ol>
        </div>
      </section>
    `;
    bindFilters();
    bindRecipeTabs();
  }

  function renderDetail(page) {
    const app = $("#app");
    document.title = `${page.title} · OPD 图像生成研究站`;
    app.innerHTML = `
      <section class="hero detail-hero">
        <div class="hero-copy">
          <div class="eyebrow"><span></span> ${escapeHtml(page.eyebrow)}</div>
          <h1>${escapeHtml(page.title)}</h1>
          <p class="lead">${escapeHtml(page.subtitle)}</p>
          <div class="hero-actions">
            <a class="button primary" href="#recipe">查看配方</a>
            <a class="button secondary" href="index.html#directions">返回研究地图</a>
          </div>
        </div>
        <aside class="mechanism-card" aria-label="机制摘要">
          <div class="trajectory-visual">
            <div class="trajectory-row">
              <span>student rollout</span>
              <i></i><i></i><i></i><i></i>
            </div>
            <div class="trajectory-row teacher">
              <span>teacher guidance</span>
              <i></i><i></i><i></i><i></i>
            </div>
            <div class="trajectory-row loss">
              <span>dense update</span>
              <i></i><i></i><i></i><i></i>
            </div>
          </div>
          <p>${escapeHtml(page.goal)}</p>
        </aside>
      </section>

      <section class="section split-section">
        <div>
          <div class="eyebrow"><span></span> Core questions</div>
          <h2>这页回答哪些问题。</h2>
        </div>
        <div class="question-list">
          ${page.coreQuestions.map((q, index) => `<article><span>${String(index + 1).padStart(2, "0")}</span><p>${escapeHtml(q)}</p></article>`).join("")}
        </div>
      </section>

      <section class="dark-band">
        <div class="section">
          <div class="section-head">
            <div>
              <div class="eyebrow on-dark"><span></span> Mechanism</div>
              <h2>机制图和概念坐标。</h2>
            </div>
            <p>把论文中的 loss、teacher access 和采样策略映射到可实现的后训练模块。</p>
          </div>
          <div class="mechanism-layout">
            <div class="slider-panel">
              <label for="step-slider">trajectory step</label>
              <input id="step-slider" type="range" min="1" max="5" value="3">
              <div class="step-track" aria-hidden="true">
                <b></b><b></b><b></b><b></b><b></b>
              </div>
              <p id="step-caption">第 3 步：teacher 在学生真实 latent state 上给目标。</p>
            </div>
            <div class="concept-cloud">
              ${page.concepts.map(item => `<span>${escapeHtml(item)}</span>`).join("")}
            </div>
          </div>
          <div class="visual-list">
            ${page.visuals.map(item => `<article><h3>建议图表</h3><p>${escapeHtml(item)}</p></article>`).join("")}
          </div>
        </div>
      </section>

      <section class="section" id="papers">
        <div class="section-head">
          <div>
            <div class="eyebrow"><span></span> Paper map</div>
            <h2>本页核心论文。</h2>
          </div>
          <p>每篇保留年份、链接、OPD 信号、摘要和可信度标签。</p>
        </div>
        <div class="paper-grid">
          ${page.papers.map(paperCard).join("")}
        </div>
      </section>

      <section class="section" id="recipe">
        <div class="section-head">
          <div>
            <div class="eyebrow"><span></span> Post-training recipe</div>
            <h2>落到工程流程时这样做。</h2>
          </div>
          <p>这些步骤不是唯一实现，但足够指导第一版实验。</p>
        </div>
        <div class="check-grid">
          ${page.recipe.map((item, index) => `<article><span>${index + 1}</span><p>${escapeHtml(item)}</p></article>`).join("")}
        </div>
      </section>

      <section class="section" id="diagnostics">
        <div class="section-head">
          <div>
            <div class="eyebrow"><span></span> Diagnostics</div>
            <h2>症状、原因、修复动作。</h2>
          </div>
          <p>点击左侧症状查看对应的失败机制和建议。</p>
        </div>
        <div class="diagnostic-layout">
          <div class="diagnostic-tabs">
            ${page.diagnostics.map((item, index) => `<button class="${index === 0 ? "active" : ""}" type="button" data-diag="${index}">${escapeHtml(item[0])}</button>`).join("")}
          </div>
          <article class="diagnostic-panel" id="diagnostic-panel" aria-live="polite"></article>
        </div>
      </section>

      <section class="section source-section">
        <div class="callout">
          <h2>验收 checklist</h2>
          <ul>${page.acceptance.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>
        <div class="source-list">
          <h3>来源</h3>
          <ol>${sourceList()}</ol>
        </div>
      </section>
    `;
    bindSlider();
    bindDiagnostics(page);
  }

  function bindFilters() {
    const buttons = $$(".filter-button");
    const cards = $$(".js-filter-card");
    buttons.forEach(button => {
      button.addEventListener("click", () => {
        const filter = button.dataset.filter;
        buttons.forEach(item => item.classList.toggle("active", item === button));
        cards.forEach(card => {
          const visible = filter === "all" || card.dataset.kind === filter;
          card.hidden = !visible;
        });
      });
    });
  }

  function bindRecipeTabs() {
    const panel = $("#recipe-panel");
    const buttons = $$(".recipe-tab");
    const render = id => {
      const recipe = recipes.find(item => item.id === id) || recipes[0];
      panel.innerHTML = `
        <h3>${escapeHtml(recipe.title)}</h3>
        <ol>${recipe.steps.map(step => `<li>${escapeHtml(step)}</li>`).join("")}</ol>
      `;
    };
    buttons.forEach(button => {
      button.addEventListener("click", () => {
        buttons.forEach(item => item.classList.toggle("active", item === button));
        render(button.dataset.recipe);
      });
    });
    render(recipes[0].id);
  }

  function bindSlider() {
    const slider = $("#step-slider");
    const caption = $("#step-caption");
    if (!slider || !caption) return;
    const copy = {
      1: "第 1 步：学生从部署时真实噪声或 mask 状态出发。",
      2: "第 2 步：学生 policy 产生自己的下一段 latent trajectory。",
      3: "第 3 步：teacher 在学生真实 latent state 上给目标。",
      4: "第 4 步：loss 对齐 velocity、mean、score 或分布。",
      5: "第 5 步：刷新 rollout，避免训练状态继续变旧。"
    };
    slider.addEventListener("input", () => {
      caption.textContent = copy[slider.value];
      $$(".step-track b").forEach((dot, index) => {
        dot.classList.toggle("active", index < Number(slider.value));
      });
    });
    slider.dispatchEvent(new Event("input"));
  }

  function bindDiagnostics(page) {
    const panel = $("#diagnostic-panel");
    const buttons = $$(".diagnostic-tabs button");
    const render = index => {
      const item = page.diagnostics[index] || page.diagnostics[0];
      panel.innerHTML = `
        <h3>${escapeHtml(item[0])}</h3>
        <dl>
          <div><dt>可能原因</dt><dd>${escapeHtml(item[1])}</dd></div>
          <div><dt>修复动作</dt><dd>${escapeHtml(item[2])}</dd></div>
        </dl>
      `;
    };
    buttons.forEach(button => {
      button.addEventListener("click", () => {
        buttons.forEach(item => item.classList.toggle("active", item === button));
        render(Number(button.dataset.diag));
      });
    });
    render(0);
  }

  function init() {
    const pageId = document.body.dataset.page || "home";
    if (pageId === "home") {
      renderHome();
      return;
    }
    const page = data.pages.find(item => item.slug === pageId);
    if (page) renderDetail(page);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
