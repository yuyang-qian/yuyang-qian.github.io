# Profile design

The README uses native text and directly clickable affiliation, project, and contact links on GitHub's normal solid background. Only the small logos are SVG images. There is no full-card image, link disclosure section, square grid, gradient, or animation. The separate dynamic page lives in `site/index.html`, with the full blue square wave field, pointer response, and per-letter keyword gradients restored from revision `9894800`. The restored design now includes mobile indentation and project separator wrapping adjustments described below.

## Editing and preview

Edit `profile.json`, add any original logos under `assets/logos/`, and run:

```sh
python3 scripts/build_profile.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open **http://127.0.0.1:8765/readme-preview.html** to review the README markup and its light/dark logo selection before pushing. The dynamic page is at **http://127.0.0.1:8765/site/**. The root `index.html` redirects there. It also opens directly from disk, and the `site/` directory contains its own display assets. See [EDITING.md](EDITING.md) for content and logo configuration.

- `scripts/profile_config.py`: validation and defaults.
- `scripts/profile_assets.py`: self-contained logo SVGs and transparency filters.
- `scripts/build_profile.py`: README, standalone page, and preview generation.
- `site/index.template.html` and `site/style.css`: standalone layout and typography.
- `site/main.js`: dynamic theme selection, square wave animation, and pointer tracking.

To rebuild the dynamic page without touching README, its preview, or its logo assets, run `python3 scripts/build_profile.py --site-only`. Site logo assets are generated separately under `site/assets/`; the protected README files are not rewritten by this mode.

New projects and categories require no renderer edits. Missing logos produce text-only links without placeholders. No third-party Python packages, remote fonts, trackers, or badge services are required. The former card exporter and generated card assets have been removed.

## Themes and typography

README background, text, and link colors inherit GitHub's page theme. The small logo images retain light/dark variants selected by `#gh-light-mode-only` and `#gh-dark-mode-only` on their wrapping anchors' hrefs. GitHub's theme selectors follow the page setting and consult the OS only in automatic mode. The project and school text links always use the configured URL unchanged; ordinary logo links open the same destination with a theme fragment. If a configured URL already has a meaningful fragment, the icon opens its image instead, preserving the exact destination on the text link.

The README uses supported `samp` elements for monospace headings and project rows. GitHub controls their exact font and spacing; the standalone page uses Consolas with system monospace fallbacks. The introduction is regular weight. README keywords remain static and bold; dynamic page keywords are larger, heavier, and animate independently per letter through theme-specific blue palettes (4.8-second loops with 0.28-second phase offsets). The greeting's period inherits the normal text color.

School logos precede their labels, with affiliations sharing one line when space permits; project logos follow their names. README category labels have trailing padding to the longest label within the same monospace element so the first project in each row starts at the same horizontal position. README categories have no leading indentation and retain their middle-dot prefix; project links also use middle-dot separators. Projects wrap between entries on narrow screens. Optional `name_lines` controls deliberate breaks on the standalone page; the native README retains complete project names.

Dynamic page category headings are 2px larger than project names: 16px versus 14px on desktop, and 15px versus 13px at widths of 560px or less. At widths of 800px or less, category headings lose their indentation while project lists use 32px of left padding. Each separator belongs to the preceding project entry, so wrapping never leaves a middle dot at the start of a line. The last project has no trailing separator.

NJU uses the supplied transparent PNG with its original colors and clipped transparent side margins. UC San Diego uses its white wordmark in dark mode. Paper-logo transparency and dark-ink treatment remain clipped to their alpha; see `assets/logos/transparent/NOTES.md`.

## Content and asset sources

- Biography and affiliations: supplied by Yu-Yang Qian.
- NJU logo: the user-supplied `assets/logos/nju_transparent.png`. Paper logos: the corresponding originals at [Yu-Yang Qian's homepage](https://www.lamda.nju.edu.cn/qianyy/): `logos/{OnlineSPEC_Logo_LQ.jpeg,AdaFlash_Vertical_Logo_LQ.jpg,jetspec_logo_LQ.jpeg,d3LLM_logo_LQ.jpeg`. These logos remain the property of their respective owners.
- UC San Diego wordmarks: the university's [official header](https://www.ucsd.edu/), `_resources/img/logo_UCSD.png` and `_resources/img/logo_UCSD_white.png`.
- PD Survey logo: `assets/logos/pd_survey.jpg`, supplied by Yu-Yang Qian. Projects without supplied logos display only their names.
- PD Survey links to `ZinYY/Awesome-Parallel-Decoding`. OnlineRLHF links to the current `ZinYY/Online_RLHF` repository.
- Counterfactual Distillation provisionally links to [Internalizing Agency from Reflective Experience](https://arxiv.org/abs/2603.16843), the supplied LEAFE paper, whose method includes counterfactual distillation. Replace its URL in `profile.json` if a different project is intended.

The README is static. The dynamic page restores continuous background motion at 9× idle speed and 3× pointer-hover speed, including the original phase continuity, hidden-tab suspension, and theme toggle. It has no pause control.
