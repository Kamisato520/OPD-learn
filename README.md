# OPD Learn

A static HTML learning site for On-Policy Distillation (OPD), with Chinese visual notes, formula derivations, and an image-generation post-training research hub.

## Entry Point

Open `index.html` locally, or deploy this repository with GitHub Pages and visit the site root.

Main pages:

- `index.html` - overall local HTML hub.
- `awesome-on-policy-distillation/opd-visual-guide.html` - compact OPD visual guide.
- `awesome-on-policy-distillation/opd-study-guide.html` - detailed OPD study guide.
- `awesome-on-policy-distillation/opd-formula-derivation.html` - OPD formula derivations.
- `awesome-on-policy-distillation/opd-image-post-training/index.html` - image-generation OPD post-training hub.

## GitHub Pages Deployment

In GitHub:

1. Open repository **Settings**.
2. Go to **Pages**.
3. Set **Source** to `Deploy from a branch`.
4. Select branch `main` and folder `/ (root)`.
5. Save and wait for Pages to publish.

After deployment, the root page should route to every HTML page in `awesome-on-policy-distillation/`.

## Notes

- The site is static HTML/CSS/JS and does not require a build step.
- Math rendering uses the MathJax CDN on pages that need formulas.
- Local paper PDFs under `awesome-on-policy-distillation/papers/start-here/` are included so the study guide links remain valid.
- `claude/DESIGN.md` is included as the design reference used by the generated pages.
