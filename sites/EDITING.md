# 手动维护个人主页

以下命令均在 **`sites/` 目录**执行。日常只需编辑 **`profile.json`**，然后运行：

```sh
python3 scripts/build_profile.py
```

这个命令会一起更新 README、README 预览、动态页面和浅色 Logo。只需要 Python 3，无需安装第三方依赖。

**只更新动态页面、保持 README 和它的 Logo 原样时，使用：**

```sh
python3 scripts/build_profile.py --site-only
```

动态页面的 HTML、CSS、JavaScript 和显示用的 Logo 都在 `site/` 内。入口为 `site/index.html`，根目录 `index.html` 会跳转到它。

修改 `site/style.css` 或 `site/main.js` 后也请运行一次 `--site-only` 构建。生成的 HTML 会根据文件内容自动更新样式和脚本链接的版本号，避免浏览器继续使用旧缓存。线上页面需推送并完成部署后才会更新。

动态页面固定使用浅色，展示问候标题、学校、研究简介和项目分类，保留方块波纹、鼠标响应和逐字渐变。页面不显示底部联系链接。`../index_backup.html` 嵌入此卡片；嵌入时自动适配高度，论文链接定位到当前主页条目，其他外部链接在新标签页打开。

## 添加项目

在 `groups` 中找到对应分类，向它的 `projects` 数组追加一项。项目顺序就是显示顺序；论文 Logo 自动显示在项目名称右侧。

有 Logo 的项目：先把图片放进 `assets/logos/`，再添加：

```json
{
  "id": "my-project",
  "name": "My Project",
  "url": "https://github.com/ZinYY/MyProject",
  "description": "鼠标悬停时显示的说明，可不填",
  "logo": "assets/logos/my-project.jpg"
}
```

没有 Logo 的项目：

```json
{
  "id": "another-project",
  "name": "Another Project",
  "url": "https://github.com/ZinYY/AnotherProject"
}
```

省略 `logo` 或写 `"logo": null`，都只显示文字，不补默认图标、不留图标占位。`id` 在所有学校和项目中须唯一，仅使用小写字母、数字、`-`、`_`；它**无需与图片文件名一致**。

主页有对应论文时，在项目中添加 `"paper_anchor": "paper-my-project"`，并为 `../index_backup.html` 中对应的论文 `<li>` 设置 `id="paper-my-project" tabindex="-1"`。动态卡片中的项目名称及 Logo 会平滑滚动到该条目，滚动结束后目标背景播放一次 1.6 秒的`#f6f7fa → #d4e0f0 → #f6f7fa` 渐变，不显示描边，重复点击会重新播放；省略此字段时直接使用 `url`，例如 PD Survey。独立打开动态页面时，论文链接也会进入 `index_backup.html` 的对应位置。刷新主页时会清除地址栏中的 `#paper-…` 后缀；首次打开带论文锚点的链接仍可正常定位。README 继续使用 `url`。TreeLoRA 排在 RLHF & RSI 的 OnlineRLHF 后面。

如果项目链接到主页中的独立 HTML 页面，可设置 `"paper_page": "bib/to-appear.html"`（路径相对主页仓库根目录）。动态卡片的名称和 Logo 会使用该本地页面，README 仍使用 `url` 中的完整地址。d3LLM-v2 排在 Diffusion LLMs 的 d3LLM 后面，使用 `assets/logos/d3LLM-v2.jpg` 并链接到此待发表页面。

项目名称默认保持单行。需要主动换行时，可额外指定 `name_lines`：

```json
{
  "id": "counterfactual",
  "name": "Counterfactual Distillation",
  "name_lines": ["Counterfactual", "Distillation"],
  "url": "https://arxiv.org/abs/2603.16843"
}
```

各行以空格拼接后须等于 `name`，不需要输入 HTML 或反斜杠。删除 `name_lines` 即恢复单行，当前项目均未设置此字段。动态页面会按指定位置换行；README 使用完整项目名称的原生文字链接，项目之间可换行，当前名称无需拆行。分类前的 `·` 和项目之间的 `·` 会自动生成。README 分类标题不缩进，并根据最长分类名自动补齐标题后的等宽间距，使每行第一个论文名纵向对齐；学校信息排在同一行，窗口较窄时由浏览器换行。

动态页面与主页统一使用 Georgia 衬线字体：问候标题为 20px / 600 字重，学校、简介、分类和项目文字均为 16px。分类使用 700 字重，正文及项目链接为 400；窄屏下仅学校信息的文字和 Logo 随宽度缩放，两项学校信息及 `/` 始终保持一行。标题使用主页的暗红色 `#880000`，链接为 `#224b8d`，关键词按 1.15 倍字号加粗并循环红色渐变。窗口宽度不超过 800px 时，论文直接紧跟各自的分类标题排列，不再强制另起一行或对齐到统一列；放不下的论文自然换行，后续行相对分类文字缩进 24px。项目之间的 `·` 自动跟随前一篇论文，换行后不会出现在行首，最后一篇后面也不会多出分隔点。

卡片间距按压缩前的 75% 设置：正文和分类标题行高为 1.45，分类之间间隔 7.5px，项目换行间隔 6px。标题、学校、简介之间的留白以及卡片内外边距均按此比例调整；行高仅缩放文字之外的额外留白。

嵌入主页时，宽屏保持与主页标题、正文和方形列表标记对齐。800px 及以下改为卡片内部对齐：问候标题、NJU Logo 和简介左对齐，左右内边距为 24px；分类列表保留方形标记，列表缩进为 16px、条目内边距为 8px。700px 及以下，卡片两侧仍保留 6px 外侧空隙。主页始终保留 520px 最小宽度。卡片背景向页面两侧留白延伸，桌面为 16px，700px 及以下为 10px；调整这些数值时需同步修改 `../index_backup.html` 和 `site/style.css` 的嵌入样式。

## Logo 去白底

- JPG/JPEG：默认用 SVG 滤镜去白底，保留原始图片文件。
- PNG、WebP、SVG：默认保留其透明度。白底 PNG 可显式设置 `remove_white: true`。
- 图片缺失、路径错误或配置不合法时，构建会显示具体字段和原因。

需要自定义时，把 `logo` 路径改成对象：

```json
{
  "src": "assets/logos/my-project.png",
  "remove_white": true,
  "height": 18,
  "canvas": [50, 24]
}
```

| 字段 | 用途 |
| --- | --- |
| `src` | 原图路径，相对 `sites/` 目录 |
| `remove_white` | 是否去白底；透明原图通常设为 `false` |
| `height` | 最终显示高度，项目默认 18px；学校可单独指定 |
| `canvas` | SVG 的宽高比例，默认 `[50, 24]`；方形 Logo 可用 `[24, 24]` |
| `fit` | 默认 `contain` 完整显示；`cover` 填满画布并裁掉超出的边缘，适合两侧有多余透明留白的图片 |

例如 UCSD 使用透明背景的官方 Logo：

```json
{
  "src": "assets/logos/ucsd.png",
  "height": 20.16,
  "canvas": [90, 20]
}
```

去白底适用于这种纯白底 Logo。若白色本身是需要保留的图案内容，请使用已经抠好的透明 PNG，并设置 `remove_white: false`。

## 添加分类、学校或联系链接

新增分类：在 `groups` 中添加 `{"name": "New Direction", "projects": [...]}`，再放入项目。空分类不会显示。

学校和身份在 `affiliations` 中维护，Logo 同样可选：

```json
{
  "id": "new-school",
  "label": "Visiting Researcher @ New School",
  "url": "https://example.edu/",
  "logo": "assets/logos/new-school.png"
}
```

README 的联系链接在 `links` 中维护：`{"label": "Blog", "url": "https://example.com/"}`。动态页面不显示此行。

`name`、`homepage`、`tagline`、`intro` 分别控制姓名、个人主页地址、页面元信息和简介。`intro_highlights` 是需要强调的短语列表。README 输出静态加粗文字；动态页面会放大关键词，并逐字循环红色渐变，其他简介文字保持常规字重。动态样式见 `site/style.css` 的 `.intro em`、`.keyword-letter`。

README 保持纯色静态样式。动态页面保留整张方块波纹、鼠标响应和逐字渐变：默认方块速度为原始速度的 9 倍，悬停时为 3 倍；关键词每 4.8 秒循环一次，相邻字母错开 0.28 秒。卡片底色为接近白色的 `#fdfeff`，方块及鼠标波纹透明度为原来的 60%。

## 检查与预览

只检查配置、不生成文件：

```sh
python3 scripts/build_profile.py --check
```

生成后启动本地预览：

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

推送前先打开 **<http://127.0.0.1:8765/readme-preview.html>**：这里以浅色展示 README 的原生文字、链接和 Logo；缩窄窗口可检查手机排版。

**<http://127.0.0.1:8765/site/>** 是浅色动态页面，支持方块波纹和鼠标交互。也可直接用浏览器打开 `site/index.html`。README 与动态页面中的学校和项目链接都可直接点击。

README 的背景和文字颜色由 GitHub 自身主题控制。Logo 统一使用浅色资源，文字和 Logo 链接始终使用 `profile.json` 中的原始 URL。

使用 `--site-only` 时，只需提交动态页面及相关代码修改，README 和它的资源不会被重写。普通完整构建则需要一起提交 README、预览和生成的资源。GitHub 最终效果以推送后的个人主页为准。

## 文件分工

| 文件 | 是否需要手动编辑 |
| --- | --- |
| `profile.json` | **日常编辑这里**：内容、链接、Logo、分类 |
| `assets/logos/` | 在这里添加原始图片 |
| `site/style.css` | 动态页面的字号、间距和动画样式；学校 Logo 与文字使用 `.affiliation` 的居中对齐 |
| `site/index.template.html` | 想改变动态页面结构时再编辑 |
| `scripts/profile_config.py` | 配置校验与默认值，日常无需改 |
| `scripts/profile_assets.py` | 通用 SVG 和去白底，日常无需改 |
| `scripts/build_profile.py` | 统一生成 README、预览和 HTML，日常无需改 |
| `README.md`、`index.html`、`readme-preview.html`、`site/index.html`、`site/assets/`、`assets/logos/themed/` | 自动生成，下次构建会覆盖，不要在这些文件里维护内容 |

JSON 不支持注释和最后一项后的多余逗号。修改后先运行 `--check`，可快速定位格式问题。
