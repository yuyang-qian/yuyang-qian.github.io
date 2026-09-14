"""Generate self-contained light logo SVGs from profile data."""

import base64
from html import escape

def svg(width, height, title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
<title id="title">{escape(title)}</title>
{body}
</svg>\n'''


def white_removal():
    """Key near-white in all RGB channels, retaining saturated blue/gold/etc."""
    channels = []
    for index, name in enumerate(('r', 'g', 'b')):
        alpha = [0, 0, 0, 0, 1]
        alpha[index] = -1
        values = '0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ' + ' '.join(map(str, alpha))
        channels.append(f'<feColorMatrix in="SourceGraphic" type="matrix" values="{values}" result="{name}"/>')
    # Screen the three inverse-channel masks: A = 1 - R*G*B.
    # Composite with the original alpha so transparent letterboxing stays clear.
    return '<defs><filter id="cutout" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">' + ''.join(channels) + '''<feComposite in="r" in2="g" operator="arithmetic" k1="-1" k2="1" k3="1" k4="0" result="rg"/>
<feComposite in="rg" in2="b" operator="arithmetic" k1="-1" k2="1" k3="1" k4="0" result="mask"/>
<feComponentTransfer in="mask" result="key"><feFuncA type="linear" slope="4" intercept="-0.18"/></feComponentTransfer>
<feComposite in="SourceGraphic" in2="key" operator="in"/></filter></defs>'''


def logo_asset(item):
    logo = item['logo']
    source = logo['_src_path']
    mime = logo['_src_mime']
    data = base64.b64encode(source.read_bytes()).decode()
    width, height = logo['canvas']
    cutout = white_removal() if logo['remove_white'] else ''
    cutout_effect = ' filter="url(#cutout)"' if cutout else ''
    fit = 'slice' if logo['fit'] == 'cover' else 'meet'
    body = f'{cutout}<g><image x="1" y="1" width="{width-2}" height="{height-2}" preserveAspectRatio="xMidYMid {fit}" xlink:href="data:{mime};base64,{data}"{cutout_effect}/></g>'
    return svg(width, height, item.get('name', item.get('label', item['id'])), body)


def build_assets(profile, destination):
    """No project names or file-name conventions are hardcoded here."""
    files = {}
    items = list(profile['affiliations'])
    items.extend(project for group in profile['groups'] for project in group['projects'])
    for item in items:
        if item['logo'] is not None:
            files[f'assets/logos/themed/{item["id"]}-light.svg'] = logo_asset(item)
    for name, contents in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding='utf-8')
    return files
