# Contributing

Contributions welcome! Open a PR to add papers, reports, or tools related to on-policy distillation.

## Inclusion criteria

A paper must pass **one** of these two tests:

1. **Student rollouts are central to the learning signal.** The student generates its own outputs, and the teacher (external, self, or privileged) provides supervision on those outputs. Student rollouts must be the primary training substrate, not incidental to the method.
2. **The work directly enables OPD deployment.** Cross-tokenizer methods, training frameworks, surveys, and tooling that make OPD practical.

### What does NOT qualify

- **Survey classification alone.** "The survey classifies it as OPD" is not sufficient. The paper's own mechanism must pass the test above.
- **Near-policy is not on-policy.** Methods using historical checkpoints or past model versions generate from a different policy than the one being trained. This is near-policy, not on-policy.
- **Self-play is not distillation.** Adversarial self-play (e.g., SPIN) involves student rollouts but lacks teacher-to-student knowledge transfer. It may fit in Precursors but not in Core OPD.
- **Reframing is not enough.** Casting alignment, imitation learning, or RL as "policy distillation" in the abstract does not make student rollouts operationally central. Check the actual training loop.
- **Domain-specific applications** need a transferable methodological contribution beyond the domain. A paper that applies vanilla OPD to a new dataset does not clear the bar.

### When in doubt

Read the actual paper. One-line descriptions and survey classifications are not sufficient for borderline decisions. If the verdict depends on whether the student generates its own training data in the core training loop, check the method section.

## Modality fairness

Apply the same bar to multimodal, speech, agent, and embodied papers as to text-only ones. If a speech-LLM paper uses student on-policy rollouts with token-level teacher feedback, it clears the same criterion as a text-only paper with the same structure. Do not apply an implicit "narrowness" penalty to non-text modalities.

## Section placement

### Core OPD Papers (~21 canonical entries)

Only add here if the paper's primary contribution is a new OPD method and it would be cited as a defining work in the field. When unsure, place in Adjacent.

### Adjacent and Enabling Work

The default landing zone. Subsections:

- **Cross-Tokenizer and Model-Family Enablers** — vocabulary alignment, cross-architecture methods
- **Mismatch Mitigation and Student Quality** — distribution gap, adaptive mixing, student output quality
- **Preference, Reward-Guided, and Hybrid RL+KD** — preference optimization, reward-guided distillation, RL+KD unification
- **Self-Play and Iterative Bootstrapping** — SPIN-style self-play, iterative best-of-N, MCTS-guided self-evolution
- **Agent, Multimodal, and Other Extensions** — agents, VLMs, speech, embodied, other modalities
- **Speculative Decoding (Draft-Model Training)** — DistillSpec, EAGLE, HASS families; draft trained from target supervision on its own rollouts
- **Precursors** — historical antecedents that predate the OPD label

### Technical Reports

Production systems using OPD as a post-training stage. Table format.

### Frameworks and Tools

Training frameworks and code. Table or bullet format depending on subsection.

## Entry format

All entries are single-line. The title is the hyperlink; the year is italicized; the description follows an em dash:

```
- [Full Paper Title](url) *(Year)* — One-line description.
```

For arxiv papers, link directly to `https://arxiv.org/abs/XXXX.XXXXX`. For blog posts, essays, model cards, or repos, use the canonical URL.

If the entry has an associated code release, append a Code badge at the end:

```
- [Title](arxiv-url) *(Year)* — Description. [![Code](https://img.shields.io/badge/Code-181717?style=flat&logo=github&logoColor=white)](repo-url)
```

For implementation repos under `### Implementations`, omit the year: `- [Repo Name](url) — Description.`

**Tables** (Technical Reports, Training Frameworks) follow the column layout already in each section — match the existing rows.

The description must be a single sentence, **hard cap 22 words** (well-written entries average 12–16). State **one mechanism phrase plus one differentiator** — how the method works at the highest level, then what makes it distinct from the rest of the list. If you need a second mechanism clause to explain the first, the first was wrong. Forbidden: equation symbols, Greek letters, benchmark numbers, hyperparameters, model/dataset enumerations, ablation lists, percentage-point gains, and any second mechanism clause introduced by a semicolon or "and." **Deletion test before committing:** at each comma, semicolon, or "and" in your draft, try deleting the trailing clause; if the entry still tells a scanning reader why to click, that clause was bloat. Look at existing entries for calibration; if your draft is longer than the longest entry in the same subsection, restart.

## Batch additions

When adding multiple papers at once:

- **Flag section growth.** If a single update would double any subsection's size, consider whether the section needs splitting or whether some entries are marginal.
- **Prioritize gap-filling over completeness.** A paper that opens a new niche (first speech OPD, first embodied OPD) has higher priority than a fourth variant in a well-covered area.
- **Cap awareness.** Large batches dilute curation signal. Prefer 5-8 high-confidence additions over 12+ with several borderline entries.

## Taxonomy and reading path

- Update the taxonomy tables only for Core OPD papers or papers that clearly extend an existing taxonomy row.
- The Start Here reading path changes rarely. Only add a paper if it is the best introduction to a topic currently underrepresented in the path.
