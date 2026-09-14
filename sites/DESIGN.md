# Profile design

The README uses native text and directly clickable affiliation, project, and contact links on GitHub's normal solid background. Only the small logos are SVG images. There is no full-card image, link disclosure section, square grid, gradient, or animation. The separate dynamic page lives in `site/index.html`, with the full blue square wave field, pointer response, and per-letter keyword gradients restored from revision `9894800`. The restored design now includes mobile indentation and project separator wrapping adjustments described below.

## Editing and preview

From the `sites/` directory, edit `profile.json`, add any original logos under `assets/logos/`, and run:

```sh
python3 scripts/build_profile.py
python3 -m http.server 8765 --bind 127.0.0.1
```

Open **http://127.0.0.1:8765/readme-preview.html** to review the README markup and its light logos before pushing. The dynamic page is at **http://127.0.0.1:8765/site/**. The root `index.html` redirects there. It also opens directly from disk, and the `site/` directory contains its own display assets. See [EDITING.md](EDITING.md) for content and logo configuration.

- `scripts/profile_config.py`: validation and defaults.
- `scripts/profile_assets.py`: self-contained logo SVGs and transparency filters.
- `scripts/build_profile.py`: README, standalone page, and preview generation.
- `site/index.template.html` and `site/style.css`: standalone layout and typography.
- `site/main.js`: square wave animation, pointer tracking, and embedded card sizing.

To rebuild the dynamic page without touching README, its preview, or its logo assets, run `python3 scripts/build_profile.py --site-only`. Site logo assets are generated separately under `site/assets/`; the protected README files are not rewritten by this mode.

New projects and categories require no renderer edits. Missing logos produce text-only links without placeholders. No third-party Python packages, remote fonts, trackers, or badge services are required. The former card exporter and generated card assets have been removed.

## Appearance and typography

The dynamic page and README preview always use a light palette. Logo generation produces only light SVGs. README text and logo links use the configured URL unchanged. README background, text, and link colors on GitHub inherit GitHub's page theme.

The dynamic card starts with a greeting heading and affiliations, followed by the research introduction and project groups. It has no theme control or contact footer. `../index_backup.html` embeds the card, which reports its height to the parent. Projects with a `paper_anchor` smoothly scroll to the matching paper in the current parent document, then update the URL fragment, focus the item, and play a single 1.6-second `#f6f7fa` → `#d4e0f0` → `#f6f7fa` background fade without an outline. Repeated clicks restart the fade. Refreshing the homepage clears a `#paper-…` fragment with `history.replaceState`; initial visits to shared paper links still navigate to the paper, and other fragments are preserved. Native links to `../../index_backup.html#paper-…` also work outside the iframe. Projects without a paper anchor, such as PD Survey, retain their external URLs and open in new tabs when embedded. TreeLoRA follows OnlineRLHF. The introduction highlights “Efficiency” and “Infra”. The background is near-white (`#fdfeff`), and square-grid and pointer-wave opacity are scaled to 60% of the original values.

The README uses supported `samp` elements for monospace headings and project rows. GitHub controls their exact font and spacing. The dynamic page uses Georgia, serif to match `index_backup.html`: its greeting is 20px with weight 600, while affiliations, introduction, category labels, and project links are 16px. Category labels are bold; body text and links have normal weight. The greeting uses the homepage's burgundy (`#880000`), and links use its blue (`#224b8d`). Dynamic keywords contain one copy of their text for selection and assistive technology. They use the same 1.15em size and bold weight as the homepage highlights, animating per letter through a red palette without extra letter spacing or text stroke (4.8-second loops with 0.28-second phase offsets).

School logos precede their labels, with affiliations sharing one line when space permits; project logos follow their names. Dynamic category headings end with a colon. On wide screens, the paper column uses the longest category heading’s measured text width plus a 4px space, keeping all first paper links aligned without an oversized fixed column. README category labels have trailing padding to the longest label within the same monospace element so the first project in each row starts at the same horizontal position. README categories have no leading indentation and retain their middle-dot prefix; project links also use middle-dot separators. Projects wrap between entries on narrow screens. Optional `name_lines` controls deliberate breaks on the standalone page; the native README retains complete project names.

On screens of 700px or less, the affiliation row keeps both full labels, logos, and the slash on one line. Its font size adapts to the available width, with logo heights scaling proportionally. Other text keeps its existing font size. At widths of 800px or less, projects follow their category heading in the same inline flow instead of a separate column. Continuation lines have a 24px hanging indent relative to the category text. Each separator belongs to the preceding project entry, so wrapping never leaves a middle dot at the start of a line. The last project has no trailing separator.

Card spacing is set to 75% of its original values: introduction and category headings have 1.45 line height, research groups have 7.5px gaps, and wrapped project rows have 6px gaps. Margins and padding use the same scale; line heights scale only the extra space beyond the text size.

On wide screens, embedding aligns the greeting with the host's section headings. Affiliations, the introduction, and the Supervisor paragraph share a 24px text indent. Research categories use native square list markers with the same 40px list indentation and 8px item padding as the paper entries. The iframe extends into the host gutters (16px, or 10px at widths of 700px and below), keeping the card border outside the aligned text. The iframe spans the host's full width including these gutters, so both documents cross the 700px breakpoint together. The host retains a 520px minimum width even on narrow screens. At 700px and below, the card keeps 6px of outside space on each side within that layout. At 800px and below, the card uses 8px of horizontal inner padding. The affiliation row begins at the same left edge as the introduction, aligned with the host’s Professor paragraph, and the native square markers align with the host’s paper-list markers. Their offsets account for the host gutter, card inset, border, and inner padding. Keep the host gutter values and the `.embedded` CSS rules in sync.

NJU uses the supplied transparent PNG with its original colors and clipped transparent side margins. UC San Diego uses its original colored wordmark. Paper-logo transparency remains clipped to its alpha; see `assets/logos/transparent/NOTES.md`.

## Content and asset sources

- Biography and affiliations: supplied by Yu-Yang Qian.
- NJU logo: the user-supplied `assets/logos/nju_transparent.png`. Paper logos: the corresponding originals at [Yu-Yang Qian's homepage](https://www.lamda.nju.edu.cn/qianyy/): `logos/{OnlineSPEC_Logo_LQ.jpeg,AdaFlash_Vertical_Logo_LQ.jpg,jetspec_logo_LQ.jpeg,d3LLM_logo_LQ.jpeg`. These logos remain the property of their respective owners.
- UC San Diego wordmark: the university's [official header](https://www.ucsd.edu/), `_resources/img/logo_UCSD.png`.
- PD Survey logo: `assets/logos/pd_survey.jpg`, supplied by Yu-Yang Qian. Projects without supplied logos display only their names.
- PD Survey links to `ZinYY/Awesome-Parallel-Decoding`. OnlineRLHF links to the current `ZinYY/Online_RLHF` repository.
- Counterfactual Distillation provisionally links to [Internalizing Agency from Reflective Experience](https://arxiv.org/abs/2603.16843), the supplied LEAFE paper, whose method includes counterfactual distillation. Replace its URL in `profile.json` if a different project is intended.

The README is static. The dynamic page restores continuous background motion at 9× idle speed and 3× pointer-hover speed, including the original phase continuity and hidden-tab suspension. It has no pause control.
