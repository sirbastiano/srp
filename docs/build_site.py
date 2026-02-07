#!/usr/bin/env python3
"""
build_site.py  –  Markdown → HTML static-site generator for sarpyx docs.

Usage:
    python build_site.py                 # default: docs/ → _site/
    python build_site.py -o /tmp/site    # custom output dir

Requirements (in the venv):
    pip install markdown pygments        # or: uv pip install markdown pygments

The generated site is ready for GitHub Pages: just push ``_site/`` as the
``gh-pages`` branch (or point Pages at the folder).
"""

from __future__ import annotations

import argparse
import re
import shutil
import textwrap
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension
from pygments import highlight as _pygments_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, TextLexer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # srp/docs/
LOGO_SRC   = SCRIPT_DIR.parent / "src" / "sarpyx_logo.png"
DEFAULT_OUT = SCRIPT_DIR / "_site"

# ---------------------------------------------------------------------------
# Navigation tree  (section title → list of (label, relative-md-path) )
# ---------------------------------------------------------------------------
NAV: list[tuple[str, list[tuple[str, str]]]] = [
    ("Home", [
        ("Introduction", "README.md"),
    ]),
    ("User Guide", [
        ("Overview",              "user_guide/README.md"),
        ("Installation",          "user_guide/installation.md"),
        ("Getting Started",       "user_guide/getting_started.md"),
        ("Basic Concepts",        "user_guide/basic_concepts.md"),
        ("Data Formats",          "user_guide/data_formats.md"),
        ("SNAP Integration",      "user_guide/snap_integration.md"),
        ("Processing Workflows",  "user_guide/processing_workflows.md"),
        ("Science Applications",  "user_guide/science_applications.md"),
        ("Troubleshooting",       "user_guide/troubleshooting.md"),
        ("Install S1-ISP",        "user_guide/INSTALL_S1ISP.md"),
    ]),
    ("Tutorials", [
        ("Overview",              "tutorials/README.md"),
        ("01 – Sub-Look Analysis","tutorials/01_first_sublook_analysis.md"),
        ("02 – SNAP Basics",      "tutorials/02_snap_integration_basics.md"),
        ("03 – Visualisation",    "tutorials/03_visualization_quality.md"),
        ("04 – Multi-temporal",   "tutorials/04_multitemporal_analysis.md"),
        ("05 – Polarimetry",     "tutorials/05_polarimetric_analysis.md"),
        ("06 – Custom Workflows", "tutorials/06_custom_workflows.md"),
        ("07 – Ship Detection",   "tutorials/07_ship_detection.md"),
        ("08 – InSAR",            "tutorials/08_interferometric_analysis.md"),
    ]),
    ("API Reference", [
        ("Overview",           "api/README.md"),
        ("Snapflow",           "api/snapflow/README.md"),
        ("Snapflow Examples",  "api/snapflow/Snapflow_usage_example.md"),
        ("Science",            "api/science/README.md"),
        ("SLA",                "api/sla/README.md"),
        ("Processor",          "api/processor/README.md"),
        ("Utils",              "api/utils/README.md"),
        ("DEM Utilities",      "api/utils/dem_utils.md"),
    ]),
    ("Examples", [
        ("Overview",           "examples/README.md"),
    ]),
    ("Developer Guide", [
        ("Architecture",       "developer_guide/architecture.md"),
        ("Contributing",       "developer_guide/contributing.md"),
    ]),
    ("Reference", [
        ("MetaParams",         "metaParams.md"),
    ]),
]


def _md_to_html_path(md_rel: str) -> str:
    """README.md → index.html,  foo/bar.md → foo/bar.html"""
    p = Path(md_rel)
    name = "index.html" if p.stem.upper() == "README" else p.with_suffix(".html").name
    return str(p.parent / name) if str(p.parent) != "." else name


# ---------------------------------------------------------------------------
# CSS + Pygments
# ---------------------------------------------------------------------------
def _build_css() -> str:
    """Return the full CSS string (site theme + Pygments code highlighting)."""
    pygments_css = HtmlFormatter(style="monokai").get_style_defs(".codehilite")
    return textwrap.dedent("""\
    /* ── Reset ────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── Palette ───────────────────────────────────────── */
    :root {
      --bg:        #0d1117;
      --bg-side:   #161b22;
      --bg-card:   #161b22;
      --border:    #30363d;
      --text:      #c9d1d9;
      --text-dim:  #8b949e;
      --accent:    #58a6ff;
      --accent2:   #3fb950;
      --accent-bg: rgba(88,166,255,.08);
      --code-bg:   #1c2128;
      --hover:     rgba(88,166,255,.12);
      --radius:    8px;
    }

    html { scroll-behavior: smooth; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      display: flex;
      min-height: 100vh;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    .sidebar {
      width: 280px;
      min-width: 280px;
      background: var(--bg-side);
      border-right: 1px solid var(--border);
      padding: 1.5rem 0;
      position: fixed;
      top: 0; left: 0; bottom: 0;
      overflow-y: auto;
      z-index: 100;
      transition: transform .25s ease;
    }
    .sidebar .logo {
      display: block;
      margin: 0 auto 1.2rem;
      width: 200px;
      height: auto;
      filter: drop-shadow(0 0 16px rgba(88,166,255,.35));
    }
    .sidebar h3 {
      color: var(--text-dim);
      font-size: .7rem;
      letter-spacing: .12em;
      text-transform: uppercase;
      padding: .8rem 1.2rem .3rem;
    }
    .sidebar a {
      display: block;
      padding: .35rem 1.2rem .35rem 1.6rem;
      color: var(--text);
      text-decoration: none;
      font-size: .88rem;
      border-left: 3px solid transparent;
      transition: all .15s;
    }
    .sidebar a:hover, .sidebar a.active {
      background: var(--hover);
      border-left-color: var(--accent);
      color: var(--accent);
    }

    /* hamburger for mobile */
    .menu-toggle {
      display: none;
      position: fixed;
      top: 1rem; left: 1rem;
      z-index: 200;
      background: var(--bg-side);
      border: 1px solid var(--border);
      color: var(--accent);
      font-size: 1.4rem;
      padding: .3rem .6rem;
      border-radius: var(--radius);
      cursor: pointer;
    }

    /* ── Main content ──────────────────────────────────── */
    .main {
      margin-left: 280px;
      flex: 1;
      padding: 2.5rem 3rem 4rem;
      max-width: 960px;
    }

    h1, h2, h3, h4, h5, h6 { color: #e6edf3; margin-top: 2rem; margin-bottom: .6rem; }
    h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: .4rem; }
    h2 { font-size: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: .3rem; }
    h3 { font-size: 1.2rem; }
    h1:first-child { margin-top: 0; }

    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }

    p { margin: .8rem 0; }

    ul, ol { margin: .5rem 0 .5rem 1.5rem; }
    li { margin: .25rem 0; }

    blockquote {
      border-left: 4px solid var(--accent);
      background: var(--accent-bg);
      padding: .8rem 1rem;
      margin: 1rem 0;
      border-radius: 0 var(--radius) var(--radius) 0;
    }

    code {
      font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
      background: var(--code-bg);
      padding: .15rem .35rem;
      border-radius: 4px;
      font-size: .88em;
    }
    pre {
      background: var(--code-bg) !important;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.2rem;
      overflow-x: auto;
      margin: 1rem 0;
      line-height: 1.5;
    }
    pre code { background: none; padding: 0; }

    /* tables */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: .9rem;
    }
    th, td {
      padding: .55rem .8rem;
      border: 1px solid var(--border);
      text-align: left;
    }
    th { background: var(--bg-side); color: #e6edf3; font-weight: 600; }
    tr:nth-child(even) { background: rgba(255,255,255,.02); }

    /* misc */
    hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
    img { max-width: 100%; height: auto; border-radius: var(--radius); }

    /* -- toc (right side) ------- */
    .toc { margin: 1rem 0 2rem; padding: 1rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); }
    .toc ul { list-style: none; margin-left: .8rem; }
    .toc > ul { margin-left: 0; }
    .toc a { font-size: .85rem; color: var(--text-dim); }
    .toc a:hover { color: var(--accent); }

    /* badges */
    .badge {
      display: inline-block;
      font-size: .72rem;
      padding: .15em .55em;
      border-radius: 999px;
      font-weight: 600;
      vertical-align: middle;
    }
    .badge-blue  { background: rgba(88,166,255,.15); color: var(--accent); }
    .badge-green { background: rgba(63,185,80,.15);  color: var(--accent2); }

    /* footer */
    .footer {
      margin-top: 4rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: .78rem;
      color: var(--text-dim);
    }

    /* ── Responsive ─────────────────────────────────── */
    @media (max-width: 800px) {
      .sidebar { transform: translateX(-100%); }
      .sidebar.open { transform: translateX(0); }
      .menu-toggle { display: block; }
      .main { margin-left: 0; padding: 1.2rem; }
    }

    /* ── Pygments overrides ────────────────────────── */
    """) + "\n" + pygments_css + "\n"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
def _render_nav_html(active_md: str, depth: int) -> str:
    root = "../" * depth if depth else "./"
    parts: list[str] = []
    for section, items in NAV:
        parts.append(f'<h3>{section}</h3>')
        for label, md_path in items:
            html_path = _md_to_html_path(md_path)
            cls = ' class="active"' if md_path == active_md else ""
            parts.append(f'<a href="{root}{html_path}"{cls}>{label}</a>')
    return "\n".join(parts)


def _page_html(title: str, body_html: str, nav_html: str, toc_html: str,
               depth: int) -> str:
    root = "../" * depth if depth else "./"
    logo = f"{root}assets/sarpyx_logo.png"
    css  = f"{root}assets/style.css"

    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="en">\n'
        f'<head>\n'
        f'  <meta charset="utf-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>{title} \u2013 sarpyx Docs</title>\n'
        f'  <link rel="stylesheet" href="{css}">\n'
        f'  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
        f'&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">\n'
        f'  <link rel="icon" href="{logo}" type="image/png">\n'
        f'</head>\n'
        f'<body>\n'
        f'  <button class="menu-toggle" onclick="document.querySelector(\'.sidebar\').classList.toggle(\'open\')">\u2630</button>\n'
        f'  <nav class="sidebar">\n'
        f'    <a href="{root}index.html"><img class="logo" src="{logo}" alt="sarpyx"></a>\n'
        f'    {nav_html}\n'
        f'  </nav>\n'
        f'  <main class="main">\n'
        f'    {body_html}\n'
        f'    <div class="footer">\n'
        f'      sarpyx Documentation &nbsp;\u00b7&nbsp; built with <code>build_site.py</code>\n'
        f'    </div>\n'
        f'  </main>\n'
        f'</body>\n'
        f'</html>\n'
    )


# ---------------------------------------------------------------------------
# Build logic
# ---------------------------------------------------------------------------
def _collect_md_files(src: Path) -> list[Path]:
    """Return all .md files in src, sorted (excluding _site output dir)."""
    return sorted(p for p in src.rglob("*.md") if "_site" not in p.parts)


def _fix_md_links(html: str, depth: int) -> str:
    """Rewrite .md hrefs → .html, fix image src paths, handle ../assets/."""
    root = "../" * depth if depth else "./"

    def _repl(m: re.Match) -> str:
        pre, href, post = m.group(1), m.group(2), m.group(3)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        # Rewrite ../assets/ → site assets/
        if "../assets/" in href or href.startswith("../src/"):
            filename = href.rsplit("/", 1)[-1]
            return f'{pre}{root}assets/{filename}{post}'
        href = re.sub(r'README\.md', 'index.html', href)
        href = re.sub(r'\.md(?=#|$)', '.html', href)
        return f'{pre}{href}{post}'
    return re.sub(r'((?:href|src)=["\'])([^"\']+)(["\'])', _repl, html)


_CODE_BLOCK_RE = re.compile(
    r'<pre><code class="language-(\w+)">(.*?)</code></pre>',
    re.DOTALL,
)
_FORMATTER = HtmlFormatter(nowrap=True, style="monokai")


def _highlight_code_blocks(html: str) -> str:
    """Apply Pygments syntax highlighting to fenced code blocks."""
    def _repl(m: re.Match) -> str:
        lang = m.group(1)
        code = m.group(2)
        # Unescape HTML entities that markdown already escaped
        import html as _html
        code = _html.unescape(code)
        try:
            lexer = get_lexer_by_name(lang)
        except Exception:
            lexer = TextLexer()
        highlighted = _pygments_highlight(code, lexer, _FORMATTER)
        return f'<pre class="codehilite"><code>{highlighted}</code></pre>'
    return _CODE_BLOCK_RE.sub(_repl, html)


def build(src_dir: Path, out_dir: Path) -> None:
    """Walk *src_dir*, convert every .md → .html inside *out_dir*."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ── assets ────────────────────────────────────────
    asset_dir = out_dir / "assets"
    asset_dir.mkdir()
    (asset_dir / "style.css").write_text(_build_css(), encoding="utf-8")
    if LOGO_SRC.exists():
        shutil.copy2(LOGO_SRC, asset_dir / "sarpyx_logo.png")
    else:
        print(f"⚠  Logo not found at {LOGO_SRC}")

    # ── collect all md paths ──────────────────────────
    md_files = _collect_md_files(src_dir)
    # Build a set of relative md paths that exist
    existing_rel: set[str] = set()
    for mf in md_files:
        existing_rel.add(str(mf.relative_to(src_dir)))

    # ── markdown converter ────────────────────────────
    md = markdown.Markdown(
        extensions=[
            "markdown.extensions.fenced_code",
            "markdown.extensions.tables",
            TocExtension(permalink=False, toc_depth="1-3"),
            "markdown.extensions.sane_lists",
            "markdown.extensions.smarty",
        ],
        extension_configs={
            "markdown.extensions.fenced_code": {},
        },
        output_format="html",
    )

    converted = 0
    for md_file in md_files:
        rel = md_file.relative_to(src_dir)
        rel_str = str(rel)

        # Determine output path
        html_name = "index.html" if rel.stem.upper() == "README" else rel.with_suffix(".html").name
        html_out = out_dir / rel.parent / html_name
        html_out.parent.mkdir(parents=True, exist_ok=True)

        # Convert
        md.reset()
        source = md_file.read_text(encoding="utf-8", errors="replace")
        body = md.convert(source)
        toc  = getattr(md, "toc", "")

        # Depth for relative root path (must be computed before link rewriting)
        depth = len(rel.parent.parts)

        # Fix internal .md links → .html and image paths
        body = _fix_md_links(body, depth)

        # Syntax-highlight code blocks via Pygments
        body = _highlight_code_blocks(body)

        # Derive title from first H1 or filename
        m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else rel.stem

        nav = _render_nav_html(rel_str, depth)
        html_out.write_text(
            _page_html(title, body, nav, toc, depth),
            encoding="utf-8",
        )
        converted += 1

    # ── also copy .py example files as plain-text downloadable ────
    for py in src_dir.rglob("*.py"):
        if py.name == "build_site.py" or "_site" in py.parts:
            continue
        dest = out_dir / py.relative_to(src_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py, dest)

    print(f"✓  Built {converted} pages → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build sarpyx HTML docs from markdown.")
    ap.add_argument("-s", "--src", type=Path, default=SCRIPT_DIR,
                    help="Source markdown directory (default: same dir as this script)")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                    help="Output directory (default: _site/ next to source)")
    args = ap.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
