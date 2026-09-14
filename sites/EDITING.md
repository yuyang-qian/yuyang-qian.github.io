# 手动维护个人主页

日常只需编辑 **`profile.json`**，然后在仓库根目录运行：

```sh
python3 scripts/build_profile.py
```

这个命令会一起更新 README、README 预览、动态页面和深浅色 Logo。只需要 Python 3，无需安装第三方依赖。

**只更新动态页面、保持 README 和它的 Logo 原样时，使用：**

```sh
python3 scripts/build_profile.py --site-only
```

动态页面的 HTML、CSS、JavaScript 和显示用的 Logo 都在 `site/` 内。入口为 `site/index.html`，根目录 `index.html` 会跳转到它。

修改 `site/style.css` 后也请运行一次 `--site-only` 构建。生成的 HTML 会根据 CSS 内容自动更新样式链接的版本号，避免浏览器继续使用旧样式缓存。线上页面需推送并完成部署后才会更新。

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

动态页面分类标题比论文名大 2px：桌面端分别为 16px / 14px，窗口宽度不超过 560px 时分别为 15px / 13px。窗口宽度不超过 800px 时，分类标题不缩进，论文列表向右缩进 32px。项目之间的 `·` 自动跟随前一篇论文，换行后不会出现在行首，最后一篇后面也不会多出分隔点。

## Logo 去白底与深色适配

- JPG/JPEG：默认用 SVG 滤镜去白底，保留原始图片文件。
- PNG、WebP、SVG：默认保留其透明度。白底 PNG 可显式设置 `remove_white: true`。
- 深色主题默认提亮黑色、深蓝色笔画，保留鲜艳的蓝色和金色。
- 图片缺失、路径错误或配置不合法时，构建会显示具体字段和原因。

需要自定义时，把 `logo` 路径改成对象：

```json
{
  "src": "assets/logos/my-project.png",
  "remove_white": true,
  "height": 18,
  "canvas": [50, 24],
  "dark_style": "lift"
}
```

| 字段 | 用途 |
| --- | --- |
| `src` | 原图路径，相对仓库根目录 |
| `remove_white` | 是否去白底；透明原图通常设为 `false` |
| `height` | 最终显示高度，项目默认 18px；学校可单独指定 |
| `canvas` | SVG 的宽高比例，默认 `[50, 24]`；方形 Logo 可用 `[24, 24]` |
| `fit` | 默认 `contain` 完整显示；`cover` 填满画布并裁掉超出的边缘，适合两侧有多余透明留白的图片 |
| `dark_style` | `lift` 提亮深色笔画；`pastel` 整体调浅；`original` 保持原色 |
| `dark_src` | 可选，单独的深色版图片；设置后优先使用它，不再自动调色 |

例如 UCSD 有官方白色 Logo：

```json
{
  "src": "assets/logos/ucsd.png",
  "dark_src": "assets/logos/ucsd-white.png",
  "height": 16.8,
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

页脚链接在 `links` 中维护：`{"label": "Blog", "url": "https://example.com/"}`。

`name`、`homepage`、`tagline`、`intro` 分别控制姓名、个人主页地址、页面元信息和简介。`intro_highlights` 是需要强调的短语列表。README 输出静态加粗文字；动态页面会放大关键词，并逐字循环蓝色渐变，其他简介文字保持常规字重。动态样式见 `site/style.css` 的 `.intro em`、`.keyword-letter`。

README 保持纯色静态样式。动态页面恢复整张方块波纹、鼠标响应、逐字渐变及深浅色切换：默认方块速度为原始速度的 9 倍，悬停时为 3 倍；关键词每 4.8 秒循环一次，相邻字母错开 0.28 秒。

## 检查与预览

只检查配置、不生成文件：

```sh
python3 scripts/build_profile.py --check
```

生成后启动本地预览：

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

推送前先打开 **<http://127.0.0.1:8765/readme-preview.html>**：这里展示 README 的原生文字、链接和 Logo，可切换浅色、深色和跟随系统；缩窄窗口可检查手机排版。

**<http://127.0.0.1:8765/site/>** 是动态页面，支持方块波纹、鼠标交互和手动切换主题。也可直接用浏览器打开 `site/index.html`。README 与动态页面中的学校、项目和联系链接都可直接点击。

README 的背景和文字颜色由 GitHub 自身主题控制。只有 Logo 使用 `#gh-light-mode-only` / `#gh-dark-mode-only` 链接标记选择图片版本，避免系统深色、GitHub 浅色时选错 Logo。不要把标记改到图片 `src` 上。项目文字链接始终使用 `profile.json` 中的原始 URL。

使用 `--site-only` 时，只需提交动态页面及相关代码修改，README 和它的资源不会被重写。普通完整构建则需要一起提交 README、预览和生成的资源。GitHub 最终效果以推送后的个人主页为准。

## 文件分工

| 文件 | 是否需要手动编辑 |
| --- | --- |
| `profile.json` | **日常编辑这里**：内容、链接、Logo、分类 |
| `assets/logos/` | 在这里添加原始图片 |
| `site/style.css` | 动态页面的字号、间距和动画样式；学校 Logo 与文字使用 `.affiliation` 的居中对齐 |
| `site/index.template.html` | 想改变动态页面结构时再编辑 |
| `scripts/profile_config.py` | 配置校验与默认值，日常无需改 |
| `scripts/profile_assets.py` | 通用 SVG、去白底、主题适配，日常无需改 |
| `scripts/build_profile.py` | 统一生成 README、预览和 HTML，日常无需改 |
| `README.md`、`index.html`、`readme-preview.html`、`site/index.html`、`site/assets/`、`assets/logos/themed/` | 自动生成，下次构建会覆盖，不要在这些文件里维护内容 |

JSON 不支持注释和最后一项后的多余逗号。修改后先运行 `--check`，可快速定位格式问题。
