"""Load and validate the human-edited profile configuration; no output side effects."""

import json
import math
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlsplit


def require_text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{location}: expected non-empty text')
    return value


def require_url(value, location):
    value = require_text(value, location)
    parsed = urlsplit(value)
    if parsed.scheme not in ('https', 'http', 'mailto') or not (parsed.netloc or parsed.path):
        raise ValueError(f'{location}: expected an https://, http://, or mailto: link')
    return value


def positive_number(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{location}: expected a positive number')
    return value


def logo_config(value, root, location):
    """A path is shorthand for a logo object; null/omitted means text only."""
    if value is None:
        return None
    if isinstance(value, str):
        value = {'src': value}
    if not isinstance(value, dict):
        raise ValueError(f'{location}: expected an image path, logo object, or null')
    logo = dict(value)
    for key in ('src', 'dark_src'):
        if key == 'dark_src' and key not in logo:
            continue
        source = require_text(logo.get(key), f'{location}.{key}')
        path = (root / source).resolve()
        if not path.is_file():
            raise ValueError(f'{location}.{key}: image does not exist: {source}')
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ('image/png', 'image/jpeg', 'image/webp', 'image/svg+xml'):
            raise ValueError(f'{location}.{key}: use PNG, JPG, WebP, or SVG')
        logo[f'_{key}_path'] = path
        logo[f'_{key}_mime'] = mime
    # JPGs conventionally contain white backing; transparent formats are kept.
    logo.setdefault('remove_white', logo['_src_mime'] == 'image/jpeg')
    if not isinstance(logo['remove_white'], bool):
        raise ValueError(f'{location}.remove_white: expected true or false')
    logo.setdefault('dark_style', 'lift')
    if logo['dark_style'] not in ('lift', 'pastel', 'original'):
        raise ValueError(f'{location}.dark_style: choose lift, pastel, or original')
    logo.setdefault('fit', 'contain')
    if logo['fit'] not in ('contain', 'cover'):
        raise ValueError(f'{location}.fit: choose contain or cover')
    logo.setdefault('height', 18)
    positive_number(logo['height'], f'{location}.height')
    logo.setdefault('canvas', [50, 24])
    if not isinstance(logo['canvas'], list) or len(logo['canvas']) != 2:
        raise ValueError(f'{location}.canvas: expected [width, height]')
    for dimension in logo['canvas']:
        if positive_number(dimension, f'{location}.canvas') <= 2:
            raise ValueError(f'{location}.canvas: dimensions must exceed 2')
    return logo


def load_profile(path, root):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{path}:{error.lineno}:{error.colno}: {error.msg}') from error
    if not isinstance(data, dict):
        raise ValueError('profile: expected a JSON object')
    for field in ('name', 'tagline', 'intro'):
        require_text(data.get(field), field)
    require_url(data.get('homepage'), 'homepage')
    data.setdefault('intro_highlights', [])
    if not isinstance(data['intro_highlights'], list):
        raise ValueError('intro_highlights: expected a list of phrases')
    for phrase in data['intro_highlights']:
        require_text(phrase, 'intro_highlights')
    identifiers = set()

    def item(value, where, label):
        if not isinstance(value, dict):
            raise ValueError(f'{where}: expected an object')
        require_text(value.get(label), f'{where}.{label}')
        require_url(value.get('url'), f'{where}.url')
        identifier = require_text(value.get('id'), f'{where}.id')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', identifier):
            raise ValueError(f'{where}.id: use lowercase letters, digits, hyphens, or underscores')
        if identifier in identifiers:
            raise ValueError(f'{where}.id: duplicate id "{identifier}"')
        identifiers.add(identifier)
        value['logo'] = logo_config(value.get('logo'), root, f'{where}.logo')
        if 'name_lines' in value:
            lines = value['name_lines']
            if not isinstance(lines, list) or len(lines) < 2:
                raise ValueError(f'{where}.name_lines: expected at least two lines of text')
            for line in lines:
                require_text(line, f'{where}.name_lines')
            if ' '.join(lines) != value[label]:
                raise ValueError(f'{where}.name_lines: lines joined with spaces must match {label}')
        if 'description' in value:
            require_text(value['description'], f'{where}.description')

    for field in ('affiliations', 'groups', 'links'):
        data.setdefault(field, [])
        if not isinstance(data[field], list):
            raise ValueError(f'{field}: expected a list')
    for i, affiliation in enumerate(data['affiliations']):
        item(affiliation, f'affiliations[{i}]', 'label')
    for i, group in enumerate(data['groups']):
        where = f'groups[{i}]'
        if not isinstance(group, dict):
            raise ValueError(f'{where}: expected an object')
        require_text(group.get('name'), f'{where}.name')
        if not isinstance(group.get('projects'), list):
            raise ValueError(f'{where}.projects: expected a list')
        for j, project in enumerate(group['projects']):
            item(project, f'{where}.projects[{j}]', 'name')
    for i, link in enumerate(data['links']):
        if not isinstance(link, dict):
            raise ValueError(f'links[{i}]: expected an object')
        require_text(link.get('label'), f'links[{i}].label')
        require_url(link.get('url'), f'links[{i}].url')
    return data
