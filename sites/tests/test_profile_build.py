"""Integration checks for adding content without changing the renderer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from build_profile import build
from profile_config import load_profile


class ProfileBuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.config = self.output / 'profile.json'
        self.data = json.loads((ROOT / 'profile.json').read_text())

    def load(self):
        self.config.write_text(json.dumps(self.data))
        return load_profile(self.config, ROOT)

    def test_add_arbitrary_content_with_optional_logos(self):
        self.data['groups'].append({
            'name': 'New & Future',
            'projects': [
                {'id': 'new-id-unrelated-to-filename', 'name': 'New <Project>',
                 'name_lines': ['New', '<Project>'],
                 'url': 'https://example.com/new?a=1&b=2',
                 'logo': {'src': 'assets/logos/pd_survey.jpg', 'height': 21}},
                {'id': 'no-logo', 'name': 'Text Only',
                 'url': 'https://example.com/text'},
            ],
        })
        self.data['affiliations'].append({
            'id': 'new-school', 'label': 'Visitor @ New School',
            'url': 'https://example.edu/',
            'logo': 'assets/logos/nju.jpg',
        })
        self.data['links'].append({'label': 'Blog', 'url': 'https://example.com/blog'})
        assets = build(self.load(), self.output)
        readme = (self.output / 'README.md').read_text()
        page = (self.output / 'site/index.html').read_text()
        for document in (readme, page):
            self.assertIn('New &amp; Future', document)
            self.assertIn('new?a=1&amp;b=2', document)
            self.assertIn('Visitor @ New School', document)
            self.assertNotIn('no-logo-light.svg', document)
        self.assertIn('https://example.com/blog', readme)
        self.assertNotIn('https://example.com/blog', page)
        self.assertIn('<a href="https://example.com/text">Text&nbsp;Only</a>', readme)
        self.assertIn('<a class="project" href="https://example.com/text"><span>Text&nbsp;Only</span></a>', page)
        self.assertIn('New&nbsp;&lt;Project&gt;', readme)
        self.assertIn('New<br>&lt;Project&gt;', page)
        self.assertIn('height="21"', page)
        self.assertIn('height="21"', readme)
        for identifier in ('new-id-unrelated-to-filename', 'new-school'):
            asset = f'assets/logos/themed/{identifier}-light.svg'
            self.assertTrue((self.output / asset).is_file())
            self.assertIn('id="cutout"', assets[asset])
            self.assertIn('in="SourceGraphic" in2="key" operator="in"', assets[asset])
        self.assertFalse((self.output / 'assets/logos/themed/no-logo-light.svg').exists())

    def test_readme_is_static_and_site_is_animated_and_build_is_repeatable(self):
        profile = self.load()
        first = build(profile, self.output)
        self.assertEqual(first, build(profile, self.output))
        document = (self.output / 'README.md').read_text()
        self.assertIn('<strong>efficiency</strong>', document)
        self.assertIn('<strong>infra</strong>', document)
        self.assertNotIn('<canvas', document)
        self.assertNotIn('keyword-letter', document)
        site = (self.output / 'site/index.html').read_text()
        self.assertIn('<canvas id="wave-grid"', site)
        self.assertIn('class="keyword-letter"', site)
        self.assertIn('class="sr-only">efficiency</span>', site)
        self.assertIn('class="sr-only">infra</span>', site)
        self.assertIn('<h1 id="intro-title">', site)
        self.assertNotIn('<footer', site)
        self.assertNotIn('<button', site)
        self.assertNotIn('<source', site)
        self.assertIn('name="color-scheme" content="light"', site)
        self.assertTrue(all(name.endswith('-light.svg') for name in first))
        self.assertIn('url=site/', (self.output / 'index.html').read_text())
        for name in ('style.css', 'main.js', 'assets/favicon.svg'):
            self.assertTrue((self.output / 'site' / name).is_file())
        for svg in first.values():
            tags = {element.tag.rsplit('}', 1)[-1] for element in ET.fromstring(svg).iter()}
            self.assertTrue(tags.isdisjoint({'animate', 'animateTransform', 'script', 'foreignObject'}))

    def test_site_only_rebuild_does_not_touch_readme_or_its_assets(self):
        profile = self.load()
        build(profile, self.output)
        protected = [self.output / 'README.md', self.output / 'readme-preview.html']
        protected += list((self.output / 'assets/logos/themed').glob('*.svg'))
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected}
        profile['name'] = 'Site-only name'
        profile['affiliations'][0]['label'] = 'Site-only affiliation'
        build(profile, self.output, site_only=True)
        for path, original in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
        self.assertIn('Site-only name', (self.output / 'site/index.html').read_text())
        self.assertIn('Site-only affiliation', (self.output / 'site/assets/logos/themed/nju-light.svg').read_text())

    def test_readme_links_are_direct_and_logos_are_light(self):
        build(self.load(), self.output)
        readme = (self.output / 'README.md').read_text()
        self.assertNotIn('prefers-color-scheme', readme)
        self.assertNotIn('#gh-', readme)
        self.assertNotIn('<details>', readme)
        self.assertNotIn('assets/readme/card-', readme)
        items = self.data['affiliations'] + self.data['links']
        items += [project for group in self.data['groups'] for project in group['projects']]
        for item in items:
            self.assertIn(f'href="{item["url"]}"', readme)
            if item.get('logo'):
                self.assertIn(f'src="assets/logos/themed/{item["id"]}-light.svg"', readme)

    def test_errors_identify_the_configuration_field(self):
        cases = [
            ('id', 'nju', 'groups[0].projects[0].id: duplicate id'),
            ('logo', 'assets/logos/not-here.png', 'groups[0].projects[0].logo.src: image does not exist'),
            ('name_lines', ['Incorrect', 'Name'], 'groups[0].projects[0].name_lines: lines joined with spaces must match name'),
        ]
        original = self.data['groups'][0]['projects'][0].copy()
        for field, value, message in cases:
            with self.subTest(field=field):
                self.data['groups'][0]['projects'][0] = {**original, field: value}
                with self.assertRaises(ValueError) as raised:
                    self.load()
                self.assertIn(message, str(raised.exception))


if __name__ == '__main__':
    unittest.main()
