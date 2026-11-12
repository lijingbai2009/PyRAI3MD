#!/usr/bin/env python3
"""
======================================================================
This module scans the source tree for host, environment and personal
identifiers that must not ship with the package.

`pyrai3md/qualification/pmd_env.yaml` is the one file allowed to carry
deployment values; every other text file is expected to use a neutral
placeholder. The scan is deliberately mechanical: it reads files and
reports `path:line:rule:fragment`, never edits anything.

Two rule groups are applied:
  hard      any text file, a real value was copied out of a site config
            or a developer machine
  narrative docs and examples, a sentence points at a named platform or
            a specific machine even though no address is present

Relative words such as the local machine and the remote side are not
rules: they carry no identity once the platform and path rules pass.

Exit codes: 0 clean, 1 violations, 2 bad arguments.
======================================================================
"""

import argparse
import ipaddress
import json
import os
import re
import sys

ROOT_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WHITELIST = {
    'pyrai3md/qualification/pmd_env.yaml',
}

SKIP_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 'build', 'dist',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules',
}
SKIP_DIR_PREFIXES = ('.venv', 'tmp_', 'evidence_')
SKIP_DIR_SUFFIXES = ('.egg-info',)

DOC_EXTS = {'.md', '.markdown', '.rst', '.yaml', '.yml'}

ALLOWED_NETS = [
    ipaddress.ip_network(net) for net in (
        '127.0.0.0/8',
        '192.0.2.0/24',
        '198.51.100.0/24',
        '203.0.113.0/24',
    )
]

# Values are assembled from adjacent literals so this file does not
# match its own rules. `--selftest` proves that.
HARD_RULES = [
    ('ipv4', re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')),
    ('site-user', re.compile(r'\bli' + r'jb\b')),
    ('developer-home', re.compile(r'/Users/' + 'jingbai')),
    ('site-tree', re.compile(r'/pub' + r'lic/(?:software|home)')),
    ('developer-prefix', re.compile(r'/opt/home' + 'brew')),
    ('developer-venv', re.compile(r'\.venv-' + 'nn')),
    ('developer-lammps', re.compile(r'\.local/lammps-' + 'mpi')),
    ('site-conda', re.compile(r'\baic' + 'clab\b')),
    ('site-cluster-b', re.compile(r'\bmlc' + r'clab[\w-]*')),
    ('site-cluster-a', re.compile(r'\bHI' + r'AM(?:-HPC)?\b')),
    ('site-module', re.compile(r'compil' + r'er/(?:gcc|intel)')),
    ('site-queue', re.compile(
        r'(?:partition|slurm_partition|--partition)[^\n]{0,24}'
        + r'expr' + r'ess\b')),
    ('site-queue-yaml', re.compile(
        r'(?:slurm_)?partition:\s*' + r'normal\b')),
    ('ssh-alias', re.compile(r'\bssh\s+(?:-J\s+h\s+h2|h2)\b')),
    ('package-manager', re.compile(
        r'\bbre' + r'w install\b|\bHome' + r'brew\b|\bhome' + r'brew\b')),
    ('developer-tool-path', re.compile(r'\.qo' + r'der-cn')),
]

NARRATIVE_RULES = [
    ('deployment-host', re.compile(r'\bdeployment (?:host|machine)\b')),
    ('on-the-deployment', re.compile(r'\bon the ' + 'deployment' + r'\b')),
    ('deployment-compound', re.compile(
        r'\bdeployment (?:build|round|reading|artifact|run|output|environment)\b')),
    ('deployment-possessive', re.compile(
        r'\bthe ' + "deployment's" + r'\b')),
    ('this-mac', re.compile(r'\bthis ' + r"Mac(?:'s)?\b")),
    ('this-host', re.compile(r'\bthis ' + 'host' + r'\b')),
    ('this-machine', re.compile(r'\bthis ' + 'machine' + r'\b')),
    ('macos', re.compile(r'\bmac' + r'OS\b')),
    ('zh-deploy', re.compile(
        r'部' + r'署机|部' + r'署 build|部' + r'署路径|部'
        + r'署读数|部' + r'署轮|部' + r'署环境')),
    ('zh-this-machine', re.compile(r'这' + r'台 ?Mac|这' + r'台机器')),
    ('node-width', re.compile(r'\b\d+' + '-way node' + r'\b')),
]

# Narrative wording that is equally wrong inside a code comment.
CODE_NARRATIVE = {
    'deployment-host', 'on-the-deployment', 'deployment-compound',
    'deployment-possessive', 'zh-deploy',
}


def _skip_dir(name):
    if name in SKIP_DIRS:
        return True
    if name.startswith(SKIP_DIR_PREFIXES):
        return True
    if name.endswith(SKIP_DIR_SUFFIXES):
        return True
    return False


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def read_text(path):
    try:
        with open(path, 'rb') as handle:
            head = handle.read(8192)
            handle.seek(0)
            data = handle.read()
    except OSError:
        return None
    if b'\x00' in head:
        return None
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None


def _ip_allowed(value):
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return True
    if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
        return True
    return any(ip in net for net in ALLOWED_NETS)


def rules_for(path):
    ext = os.path.splitext(path)[1].lower()
    rules = [('hard', name, pattern) for name, pattern in HARD_RULES]
    if ext in DOC_EXTS:
        rules += [('narrative', name, pattern)
                  for name, pattern in NARRATIVE_RULES]
    elif ext == '.py':
        rules += [('narrative', name, pattern)
                  for name, pattern in NARRATIVE_RULES
                  if name in CODE_NARRATIVE]
    return rules


def scan_file(path, root):
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    text = read_text(path)
    if text is None:
        return []
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for category, name, pattern in rules_for(path):
            for match in pattern.finditer(line):
                if name == 'ipv4' and _ip_allowed(match.group(0)):
                    continue
                hits.append({
                    'path': rel,
                    'line': lineno,
                    'rule': name,
                    'category': category,
                    'fragment': line.strip()[:160],
                })
                break
    return hits


def scan(root, only_rule=None):
    hits = []
    for path in iter_files(root):
        rel = os.path.relpath(path, root).replace(os.sep, '/')
        if rel in WHITELIST:
            continue
        file_hits = scan_file(path, root)
        if only_rule:
            file_hits = [hit for hit in file_hits if hit['rule'] == only_rule]
        hits.extend(file_hits)
    return hits


def summarize(hits):
    by_rule = {}
    by_category = {'hard': 0, 'narrative': 0}
    for hit in hits:
        by_rule[hit['rule']] = by_rule.get(hit['rule'], 0) + 1
        by_category[hit['category']] += 1
    lines = {(hit['path'], hit['line']) for hit in hits}
    files = {hit['path'] for hit in hits}
    return {
        'matches': len(hits),
        'lines': len(lines),
        'files': len(files),
        'hard': by_category['hard'],
        'narrative': by_category['narrative'],
        'by_rule': dict(sorted(by_rule.items())),
    }


def print_report(hits, summary):
    for hit in hits:
        print('%s:%d:%s:%s' % (
            hit['path'], hit['line'], hit['rule'], hit['fragment']))
    print('-- summary --')
    for name, count in summary['by_rule'].items():
        print('%-24s %d' % (name, count))
    print('violations: %d matches on %d lines in %d files '
          '(hard=%d, narrative=%d)' % (
              summary['matches'], summary['lines'], summary['files'],
              summary['hard'], summary['narrative']))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Report host, environment and personal identifiers '
                    'that must not ship with the package.')
    parser.add_argument('--root', default=ROOT_DEFAULT)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--rule')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)

    if args.selftest:
        hits = scan_file(os.path.abspath(__file__),
                         os.path.dirname(os.path.abspath(__file__)))
        summary = summarize(hits)
        print_report(hits, summary)
        return 0 if not hits else 1

    if not os.path.isdir(args.root):
        print('not a directory: %s' % args.root, file=sys.stderr)
        return 2

    hits = scan(os.path.abspath(args.root), only_rule=args.rule)
    summary = summarize(hits)
    if args.json:
        print(json.dumps({'summary': summary, 'violations': hits},
                         ensure_ascii=False, indent=2))
    elif args.summary:
        print('-- summary --')
        for name, count in summary['by_rule'].items():
            print('%-24s %d' % (name, count))
        print('violations: %d matches on %d lines in %d files '
              '(hard=%d, narrative=%d)' % (
                  summary['matches'], summary['lines'], summary['files'],
                  summary['hard'], summary['narrative']))
    else:
        print_report(hits, summary)
    return 0 if not hits else 1


if __name__ == '__main__':
    sys.exit(main())