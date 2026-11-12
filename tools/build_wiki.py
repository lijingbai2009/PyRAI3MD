"""
======================================================================
Convert the PyRAI3MD manual (DOCX) into a Jekyll site for GitHub Pages.

The manual is split by functional keyword: every section of Chapter 5
(one input section or one skill) becomes its own page, while the other
chapters and the appendices each become one page. The site uses the
Just the Docs remote theme, so navigation, search and anchors are
produced by GitHub Pages at build time.

Generated pages carry YAML front matter and links point at the .html
files Jekyll emits. The DOCX remains the source of truth; rerun this
script after editing it.

Usage
-----
    python3 tools/build_wiki.py
    python3 tools/build_wiki.py --docx manual/PyRAI3MD_manual.docx --out docs

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]

CONFIG = """title: PyRAI3MD
description: User manual for PyRAI3MD 3.0.0
remote_theme: just-the-docs/just-the-docs
search_enabled: true
color_scheme: light
nav_sort: case_insensitive

markdown: kramdown
kramdown:
  input: GFM
  auto_ids: true
  hard_wrap: false

plugins:
  - jekyll-remote-theme

exclude:
  - "~$RAI3MD_manual.docx"
  - README.md
"""

APPENDIX_SLUG = {
    'a': 'appendix-a-units-and-sign-conventions',
    'b': 'appendix-b-index-bases-and-derived-quantities',
    'c': 'appendix-c-keyword-status-index',
    'd': 'appendix-d-refusal-and-limitation-matrix',
    'e': 'appendix-e-migrating-pyrai2md-inputs',
    'f': 'appendix-f-output-file-index',
    'g': 'appendix-g-troubleshooting',
    'h': 'appendix-h-glossary',
}

CHAPTER_SLUG = {
    1: '01-what-is-pyrai3md',
    2: '02-features',
    3: '03-installation',
    4: '04-input-files-and-conventions',
    5: '05-keyword-sections',
    6: '06-workflows',
    7: '07-multiscale-calculations',
    8: '08-periodic-systems',
    9: '09-constraints',
    10: '10-machine-learning-potentials',
    11: '11-parallel-and-remote-execution',
    12: '12-output-and-analysis',
    13: '13-examples-and-tutorials',
    14: '14-qualification-and-verification',
    15: '15-extending-pyrai3md',
}


# ---------------------------------------------------------------- docx model
def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, doc)
        elif child.tag.endswith('}tbl'):
            yield Table(child, doc)


def para_style(paragraph):
    try:
        return paragraph.style.name or ''
    except Exception:
        return ''


def is_heading(paragraph, level):
    return para_style(paragraph) == 'Heading %d' % level


def is_code_table(table):
    if len(table.columns) != 1:
        return False
    tc_pr = table.cell(0, 0)._tc.tcPr
    if tc_pr is None:
        return False
    shd = tc_pr.find(qn('w:shd'))
    return shd is not None and shd.get(qn('w:fill')) not in (None, 'auto')


def is_keyword_table(table):
    if len(table.columns) != 3:
        return False
    label = table.rows[0].cells[0].text.strip()
    return label.startswith('&') or label == 'all skill sections'


def is_profile_table(table):
    if len(table.columns) != 3:
        return False
    return [c.text.strip() for c in table.rows[0].cells] == [
        'Profile', 'Capability', 'Method']


def cell_text(cell):
    parts = [p.text.strip() for p in cell.paragraphs]
    return '\n'.join(part for part in parts if part)


def table_rows(table):
    return [[cell_text(cell) for cell in row.cells] for row in table.rows]


def slug(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', '-', text.strip())


def url(filename):
    return filename[:-3] + '.html' if filename.endswith('.md') else filename


def page_number(title):
    match = re.match(r'^(\d+|[A-H])\.\s', title)
    return match.group(1) if match else ''


def section_key(title):
    match = re.match(r'^(\d+)\.(\d+)\.\s', title)
    return '%s.%s' % (match.group(1), match.group(2)) if match else None


# -------------------------------------------------------------------- linking
class Linker(object):
    def __init__(self):
        self.files = {}
        self.anchors = {}

    def register_page(self, number, filename):
        self.files[number] = filename

    def register_anchor(self, number, filename, anchor):
        self.anchors[number] = (filename, anchor)

    def target(self, number, current):
        if number in self.files:
            filename, anchor = self.files[number], None
        elif number in self.anchors:
            filename, anchor = self.anchors[number]
        else:
            filename = self.files.get(number.split('.')[0])
            anchor = None
        if filename is None:
            return None
        if filename == current:
            return '#' + anchor if anchor else ''
        return url(filename) + ('#' + anchor if anchor else '')

    def apply(self, text, current):
        # The DOCX carries a few doubled connectors from an earlier build;
        # normalise them here so the site reads correctly.
        text = re.sub(r'\bSections Section (\d+\.\d+) to Section (\d+\.\d+)',
                      r'Sections \1 to \2', text)
        text = text.replace('Section Section ', 'Section ')
        text = text.replace('Appendix Section ', 'Appendix ')
        # The chapter introduction used to advertise the development documents,
        # which the site does not publish.
        text = text.replace(' and points to the development documents behind '
                            'each layer', '')

        def section(match):
            word = match.group(1)
            first = match.group(2)
            second = match.group(3)
            href = self.target(first, current)
            if href is None:
                return match.group(0)
            first_md = ('[%s %s](%s)' % (word, first, href)) if href else (
                '%s %s' % (word, first))
            if not second:
                return first_md
            href2 = self.target(second, current)
            second_md = ('[%s](%s)' % (second, href2)) if href2 else second
            return '%s to %s' % (first_md, second_md)

        text = re.sub(r'\b(Section|Sections)\s+(\d+(?:\.\d+)?)'
                      r'(?:\s+to\s+(\d+(?:\.\d+)?))?', section, text)

        def appendix(match):
            letter = match.group(1)
            href = self.target(letter, current)
            if href is None or href == '':
                return match.group(0)
            return '[Appendix %s](%s)' % (letter, href)

        return re.sub(r'\bAppendix\s+([A-H])\b', appendix, text)


# ------------------------------------------------------------------- markdown
def md_escape(text):
    if re.search(r'&[A-Za-z#][A-Za-z0-9]*;', text):
        text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    return text.replace('*', '\\*')


def cell_md(text, linker, current):
    text = md_escape(linker.apply(text, current))
    return text.replace('|', '\\|').replace('\n', '<br>')


def code_language(text):
    body = text.strip()
    if body.startswith('{') or body.startswith('['):
        return 'json'
    first = next((line.strip() for line in body.splitlines()
                  if line.strip() and not line.strip().startswith(('#', '!'))), '')
    if first.startswith('&') or first.startswith('$'):
        return 'text'
    if first.startswith(('python ', 'python3 ', 'pip ', 'bash ', 'git ', 'cd ',
                         'module ', 'export ', 'sbatch', 'srun')):
        return 'bash'
    if re.match(r'^[A-Za-z_][\w.-]*:\s', first) or first.endswith(':'):
        return 'yaml'
    return 'text'


def render_code(text):
    text = text.rstrip('\n')
    fence = '```'
    while fence in text:
        fence += '`'
    return '%s%s\n%s\n%s' % (fence, code_language(text), text, fence)


def render_table(rows, linker, current, keyword=False, profile=False):
    if not rows:
        return ''
    out = []
    if keyword:
        out += ['**%s**' % md_escape(rows[0][0].strip()), '']
        header = ['Keyword', 'Default', 'Description']
        body = rows[1:]
    elif profile:
        header, body = rows[0], rows[1:]
    else:
        header, body = rows[0], rows[1:]
    out.append('| ' + ' | '.join(cell_md(c, linker, current) for c in header) + ' |')
    out.append('| ' + ' | '.join(':---' for _ in header) + ' |')
    for row in body:
        # The site is public; the repository's development documents are not,
        # so a row that points at one (`dev/...`) is left out instead of
        # publishing a path the reader cannot open.
        if row and row[0].strip().startswith(('docs/', 'dev/')):
            continue
        padded = list(row) + [''] * (len(header) - len(row))
        out.append('| ' + ' | '.join(
            cell_md(c, linker, current) for c in padded) + ' |')
    return '\n'.join(out)


def render_paragraph(paragraph, linker, current):
    text = paragraph.text.strip()
    if not text:
        return ''
    # Introduces the development-document table; the site carries neither.
    if text.startswith('The repository carries the design documents'):
        return ''
    runs = paragraph.runs
    if runs and runs[0].bold and runs[0].text.strip().lower() == 'note.':
        rest = text[len('Note.'):].strip()
        return '> **Note.** ' + md_escape(linker.apply(rest, current))
    if runs and runs[0].bold and '\t' in text:
        keyword, rest = text.split('\t', 1)
        keyword, rest = keyword.strip(), rest.strip()
        if keyword and len(keyword) < 40 and ' ' not in keyword:
            if rest:
                return '**%s** - %s' % (md_escape(keyword),
                                        md_escape(linker.apply(rest, current)))
            return '**%s**' % md_escape(keyword)
    if text.startswith('\u2022'):
        return '- ' + md_escape(linker.apply(text.lstrip('\u2022').strip(), current))
    return md_escape(linker.apply(text, current))


class Page(object):
    def __init__(self, title, number, nav_order, parent=None, has_children=False):
        self.title = title
        self.number = number
        self.filename = ''
        self.nav_order = nav_order
        self.parent = parent
        self.has_children = has_children
        self.blocks = []
        self.children = []
        self.anchors = {}
        self.section_page = False

    def add(self, kind, payload):
        self.blocks.append((kind, payload))


def render_page(page, linker):
    out = []
    for kind, payload in page.blocks:
        if kind == 'h2':
            out.append('## ' + payload)
        elif kind == 'h3':
            out.append(('## ' if page.section_page else '### ') + payload)
        elif kind == 'code':
            out.append(render_code(payload))
        elif kind == 'table':
            out.append(payload)
        elif kind == 'para':
            rendered = render_paragraph(payload, linker, page.filename)
            if rendered:
                out.append(rendered)
    return '\n\n'.join(part for part in out if part).strip() + '\n'


def front_matter(page):
    lines = ['---', 'title: "%s"' % page.title.replace('"', '\\"'),
             'nav_order: %s' % page.nav_order]
    if page.has_children:
        lines.append('has_children: true')
    if page.parent:
        lines.append('parent: "%s"' % page.parent.replace('"', '\\"'))
    lines.append('---')
    return '\n'.join(lines) + '\n\n'


# ---------------------------------------------------------------------- main
def build(docx_path, out_dir):
    doc = Document(str(docx_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    front = []
    for block in iter_blocks(doc):
        if isinstance(block, Paragraph) and is_heading(block, 1):
            break
        front.append(block)

    citation_at = None
    for index, block in enumerate(front):
        if isinstance(block, Paragraph) and block.text.strip() == 'Citation':
            citation_at = index
            break
    title_blocks = front[:citation_at] if citation_at is not None else front
    citation_blocks = front[citation_at:] if citation_at is not None else []

    title_lines = [b.text.strip() for b in title_blocks
                   if isinstance(b, Paragraph) and b.text.strip()]
    citation_paras = [b for b in citation_blocks
                      if isinstance(b, Paragraph) and b.text.strip()
                      and not para_style(b).startswith('toc')
                      and b.text.strip() != 'Contents']

    chapters = []
    chapter = None
    page = None
    for block in iter_blocks(doc):
        if isinstance(block, Paragraph) and is_heading(block, 1) and page_number(block.text):
            chapter = Page(block.text.strip(), page_number(block.text.strip()), 0)
            chapters.append(chapter)
            page = chapter
            continue
        if chapter is None:
            continue
        if isinstance(block, Paragraph) and is_heading(block, 2):
            key = section_key(block.text.strip())
            if chapter.number == '5' and key:
                child = Page(block.text.strip(), key, 0, parent=chapter.title)
                child.section_page = True
                chapter.children.append(child)
                page = child
                continue
            page.add('h2', block.text.strip())
            if key:
                page.anchors[key] = slug(block.text.strip())
            continue
        if isinstance(block, Paragraph) and is_heading(block, 3):
            page.add('h3', block.text.strip())
            continue
        if isinstance(block, Table):
            if is_code_table(block):
                text = '\n'.join(p.text for p in block.cell(0, 0).paragraphs)
                page.add('code', text)
            elif is_keyword_table(block):
                page.add('table', (table_rows(block), 'keyword'))
            elif is_profile_table(block):
                page.add('table', (table_rows(block), 'profile'))
            else:
                page.add('table', (table_rows(block), 'plain'))
            continue
        if isinstance(block, Paragraph):
            if para_style(block).startswith('toc') or block.text.strip() == 'Contents':
                continue
            page.add('para', block)

    # -------- assign filenames
    for chapter in chapters:
        if chapter.number.isdigit():
            chapter.filename = CHAPTER_SLUG[int(chapter.number)] + '.md'
            chapter.nav_order = int(chapter.number)
            chapter.has_children = bool(chapter.children)
    appendix_container = Page('Appendices', 'appendix', 16, has_children=True)
    appendix_container.filename = 'appendices.md'

    index = 1
    for child in chapters[[c.number for c in chapters].index('5')].children:
        child.filename = '05-%02d-%s.md' % (
            index, slug(re.sub(r'^\d+\.\d+\.\s*', '', child.title)))
        child.nav_order = index
        index += 1

    for chapter in chapters:
        if not chapter.number.isdigit():
            chapter.filename = APPENDIX_SLUG[chapter.number.lower()] + '.md'
            chapter.nav_order = ord(chapter.number.lower()) - ord('a') + 1
            chapter.parent = appendix_container.title

    # -------- register everything before rendering so links resolve
    linker = Linker()
    for chapter in chapters:
        linker.register_page(chapter.number, chapter.filename)
        for key, anchor in chapter.anchors.items():
            linker.register_anchor(key, chapter.filename, anchor)
        for child in chapter.children:
            linker.register_page(child.number, child.filename)
    linker.register_page('appendix', appendix_container.filename)

    # -------- re-render tables with the linker available
    for owner in chapters + [c for ch in chapters for c in ch.children]:
        rebuilt = []
        for kind, payload in owner.blocks:
            if kind == 'table' and isinstance(payload, tuple):
                rows, style = payload
                rebuilt.append(('table', render_table(
                    rows, linker, owner.filename,
                    keyword=(style == 'keyword'),
                    profile=(style == 'profile'))))
            else:
                rebuilt.append((kind, payload))
        owner.blocks = rebuilt

    # -------- write pages
    for chapter in chapters:
        body = render_page(chapter, linker)
        if chapter.children:
            listing = ['## Sections', '']
            listing += ['- [%s](%s)' % (child.title, url(child.filename))
                        for child in chapter.children]
            body = body.rstrip() + '\n\n' + '\n'.join(listing) + '\n'
        (out_dir / chapter.filename).write_text(
            front_matter(chapter) + body, encoding='utf-8')
        for child in chapter.children:
            (out_dir / child.filename).write_text(
                front_matter(child) + render_page(child, linker), encoding='utf-8')

    lines = ['Reference material: units, index bases, keyword status, refusals, '
             'migration, output files, troubleshooting and the glossary.', '']
    lines += ['- [%s](%s)' % (c.title, url(c.filename))
              for c in chapters if not c.number.isdigit()]
    (out_dir / appendix_container.filename).write_text(
        front_matter(appendix_container) + '\n'.join(lines) + '\n',
        encoding='utf-8')

    citation = Page('Citation', 'citation', 17)
    citation.filename = 'citation.md'
    for paragraph in citation_paras[1:]:
        citation.add('para', paragraph)
    (out_dir / citation.filename).write_text(
        front_matter(citation) + render_page(citation, linker), encoding='utf-8')

    # -------- index
    index_lines = []
    image = None
    for rel in doc.part.rels.values():
        if 'image' in rel.reltype:
            image = rel.target_part
            break
    if image is not None:
        assets = out_dir / 'assets' / 'images'
        assets.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.partname).suffix or '.png'
        image_name = 'pyrai3md-logo' + suffix
        (assets / image_name).write_bytes(image.blob)
        index_lines += ['![PyRAI3MD logo](assets/images/%s)' % image_name, '']

    if title_lines:
        tagline = []
        for line in title_lines:
            if line.lower() == 'user manual' or line.startswith('Version '):
                break
            tagline.append(line)
        if tagline:
            index_lines += ['*' + ' '.join(tagline).strip() + '*', '']
        for line in title_lines:
            if line.lower() == 'user manual':
                index_lines += ['## User Manual', '']
            elif line.startswith('Version ') or line.startswith('Nov '):
                index_lines.append(line + '  ')
        index_lines += ['']
        author_done = False
        for line in title_lines:
            if line == 'Jingbai Li':
                index_lines += ['**Jingbai Li**  ']
                author_done = True
            elif author_done and ('Institute' in line or 'University' in line):
                index_lines.append(line + '  ')
        index_lines += ['']

    index_lines += ['## Contents', '']
    for chapter in chapters:
        if not chapter.number.isdigit():
            continue
        index_lines.append('- [%s](%s)' % (chapter.title, url(chapter.filename)))
        for child in chapter.children:
            index_lines.append('    - [%s](%s)' % (child.title, url(child.filename)))
    index_lines.append('- [%s](%s)' % (appendix_container.title,
                                       url(appendix_container.filename)))
    for chapter in chapters:
        if not chapter.number.isdigit():
            index_lines.append('    - [%s](%s)' % (chapter.title, url(chapter.filename)))
    index_lines.append('- [%s](%s)' % (citation.title, url(citation.filename)))
    index_lines += ['', 'The source of this site is the Word manual in '
                    '`manual/`; the site is rebuilt from it with '
                    '`tools/build_wiki.py`.']

    index_page = Page('PyRAI3MD', 'index', 0)
    index_page.filename = 'index.md'
    (out_dir / 'index.md').write_text(
        front_matter(index_page) + '\n'.join(index_lines) + '\n', encoding='utf-8')

    (out_dir / '_config.yml').write_text(CONFIG, encoding='utf-8')

    generated = {p.filename for p in chapters}
    generated |= {p.filename for p in
                  [c for ch in chapters for c in ch.children]}
    generated |= {index_page.filename, appendix_container.filename,
                  citation.filename}
    for stale in out_dir.glob('*.md'):
        if stale.name in generated:
            continue
        if re.match(r'^(\d{2}-|appendix-|appendices\.md$|index\.md$|citation\.md$)', stale.name):
            stale.unlink()

    written = len(list(out_dir.glob('*.md')))
    print('chapters %d, keyword pages %d, appendix pages %d' % (
        len([c for c in chapters if c.number.isdigit()]),
        len([c for c in chapters if c.number == '5'][0].children),
        len([c for c in chapters if not c.number.isdigit()])))
    print('wrote %d pages to %s' % (written, out_dir))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--docx', default=str(ROOT / 'manual' / 'PyRAI3MD_manual.docx'))
    parser.add_argument('--out', default=str(ROOT / 'docs'))
    args = parser.parse_args(argv)
    return build(Path(args.docx), Path(args.out))


if __name__ == '__main__':
    sys.exit(main())
