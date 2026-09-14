# Transparent logo assets

These PNG cutouts were created with the built-in imagegen tool from the original logos in the parent directory. The original files are preserved. Logo names, lettering, colors, and shapes were requested to stay unchanged.

Final PNG files: `adaflash.png`, `jetspec.png`, and `d3llm.png`.

OnlineSPEC's SVG assets in `../themed/` embed the untouched original JPEG and use an SVG color-matrix filter to make near-white pixels transparent. This preserves the original lettering and emblem geometry. No generated checkerboard is used.

NJU now uses the user-supplied `../nju_transparent.png` directly, with `remove_white: false`. Its existing alpha, white shield, and purple artwork are retained. `fit: cover` only clips excess transparent side margins in the SVG display viewport; the PNG itself is unchanged.

The white-removal filter clips its output to the original `SourceAlpha`. This keeps transparent letterboxing above and below the image transparent instead of turning it into black bars.

PD Survey uses the same SVG white-removal filter on the user-supplied `../pd_survey.jpg`, preserving its original black/gold lettering. The original UC San Diego PNG already has an alpha channel, so it does not need extraction. Projects without supplied logos have no fallback icon.

The profile builder creates one light SVG for every logo configured in `profile.json`, in `../themed/`. Logos keep their original colors and transparent negative space. No rectangular backdrop is added.

## Prompt used for each original logo

> Use case: background-extraction. Edit target: the single supplied original logo. Remove its white background and white negative spaces, producing a clean logo cutout with a real transparent alpha channel, saved as PNG. Preserve the original logo exactly: all text, spelling, lettering, shapes, emblem details, colors, relative positions, proportions and complete silhouette. This is a precise extraction, no redesign or relettering. Keep every black/navy, blue/yellow, or purple foreground stroke from the input intact. Tight framing with a small transparent margin; no clipping. No solid backing, no checkerboard drawn into pixels, no shadows, no new text. This file will be used as a tiny logo in a GitHub profile.

## Transparency correction prompt

Used for outputs that had simulated transparency instead of a real alpha channel:

> Remove the white background from this logo. Return a PNG with a REAL transparent alpha channel (RGBA, with background alpha=0). Absolutely no checkerboard pattern or simulated transparency in RGB. Preserve the original foreground and exact text. No redesign. This must be an actual transparent cutout, with smooth clean edges. The previous attempt was invalid because it painted checkerboard pixels into an RGB file; that is not transparency.
