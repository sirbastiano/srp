#!/usr/bin/env python3
"""Generate a production-ready static HTML documentation site for GitHub Pages.

This generator is intentionally source-driven:
- API data comes from docs/api_inventory_detailed.json (which is source-derived)
- Project metadata comes from pyproject.toml and repository files
- No external build tools are required to serve generated output
"""

from __future__ import annotations

import html
import json
import re
import textwrap
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
API_DIR = DOCS_DIR / "api"
ASSETS_DIR = DOCS_DIR / "assets"
API_INVENTORY_PATH = DOCS_DIR / "api_inventory_detailed.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"
README_PATH = ROOT / "README.md"


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "section"


def module_to_import(module_path: str) -> str:
    return module_path.removesuffix(".py").replace("/", ".")


def module_slug(module_path: str) -> str:
    return module_path.removesuffix(".py").replace("/", "_")


def format_default(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def parse_pyproject() -> dict[str, Any]:
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data.get("project", {})


def parse_readme_highlights() -> list[str]:
    if not README_PATH.exists():
        return []
    text = README_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_highlights = False
    highlights: list[str] = []
    for line in lines:
        if line.strip().lower() == "## highlights":
            in_highlights = True
            continue
        if in_highlights and line.startswith("## "):
            break
        if in_highlights and line.strip().startswith("- "):
            highlights.append(line.strip()[2:].strip())
    return highlights


def parse_doc_params(doc: str | None) -> dict[str, str]:
    if not doc:
        return {}
    lines = doc.splitlines()
    param_desc: dict[str, str] = {}

    # Google-style Args section
    in_args = False
    current_name: str | None = None
    buffer: list[str] = []

    def flush_current() -> None:
        nonlocal current_name, buffer
        if current_name is not None:
            txt = " ".join(s.strip() for s in buffer if s.strip())
            param_desc[current_name] = txt if txt else "inferred from implementation."
        current_name = None
        buffer = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped in {"Args:", "Arguments:", "Parameters:", "Parameters"}:
            in_args = True
            continue

        if in_args and stripped in {
            "Returns:",
            "Returns",
            "Raises:",
            "Raises",
            "Examples:",
            "Examples",
            "Notes:",
            "Note:",
            "Attributes:",
        }:
            flush_current()
            in_args = False
            continue

        if in_args:
            # Numpy-style separator under "Parameters"
            if re.fullmatch(r"-+", stripped):
                continue
            m = re.match(r"^\s*(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\)|:\s*[^:]+)?\s*:\s*(.*)$", line)
            if not m:
                m = re.match(r"^\s*(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^\n]*$", line)
                if m:
                    flush_current()
                    current_name = m.group(1)
                    buffer = []
                    continue
            if m:
                flush_current()
                current_name = m.group(1)
                first = m.group(2).strip() if len(m.groups()) > 1 else ""
                buffer = [first] if first else []
            elif current_name and (line.startswith(" " * 4) or line.startswith("\t") or line.startswith(" " * 8)):
                buffer.append(stripped)
            elif not stripped:
                continue

    flush_current()
    return param_desc


def parse_doc_section(doc: str | None, headers: set[str]) -> str | None:
    if not doc:
        return None
    lines = doc.splitlines()
    capture = False
    captured: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped in headers:
            capture = True
            continue
        if capture:
            if stripped in {
                "Args:",
                "Arguments:",
                "Parameters:",
                "Parameters",
                "Returns:",
                "Returns",
                "Raises:",
                "Raises",
                "Examples:",
                "Examples",
                "Notes:",
                "Note:",
                "Attributes:",
            }:
                break
            if re.fullmatch(r"-+", stripped):
                continue
            captured.append(raw)
    text = "\n".join(captured).strip()
    if not text:
        return None
    # Keep section concise for per-API rendering
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > 900:
        text = text[:900].rsplit(" ", 1)[0] + " ..."
    return text


def build_example(module_import: str, symbol_name: str, raw_params: list[dict[str, Any]], class_name: str | None = None) -> str:
    clean_params = [p for p in raw_params if p.get("name") not in {"self", "cls"}]
    required = [p for p in clean_params if p.get("required") is True]

    def placeholder(name: str) -> str:
        return f"<{name.lstrip('*')}>"

    args = ", ".join(f"{p['name'].lstrip('*')}={placeholder(p['name'])}" for p in required)

    if class_name:
        call = f"obj.{symbol_name}({args})" if args else f"obj.{symbol_name}()"
        return textwrap.dedent(
            f"""\
            from {module_import} import {class_name}

            # Constructor arguments are inferred from implementation.
            obj = {class_name}(...)  # inferred from implementation
            result = {call}
            """
        ).strip()

    call = f"{symbol_name}({args})" if args else f"{symbol_name}()"
    return textwrap.dedent(
        f"""\
        from {module_import} import {symbol_name}

        result = {call}
        """
    ).strip()


def infer_edge_cases(raw_params: list[dict[str, Any]], exceptions: list[str], doc: str | None) -> str:
    notes: list[str] = []
    if exceptions:
        notes.append("May raise: " + ", ".join(sorted(set(exceptions))) + ".")
    optional_count = sum(1 for p in raw_params if p.get("default") is not None)
    if optional_count:
        notes.append("Includes optional parameters with implementation-defined fallback behavior.")
    if doc and "None" in doc and "Returns" in doc:
        notes.append("Documented return may be None for some execution paths.")
    if not notes:
        notes.append("No explicit edge-case section found; behavior is inferred from implementation.")
    return " ".join(notes)


def collect_inventory() -> list[dict[str, Any]]:
    inventory = json.loads(API_INVENTORY_PATH.read_text(encoding="utf-8"))
    return [item for item in inventory if item.get("module", "").startswith("sarpyx/")]


def category_for_module(module_path: str) -> str:
    parts = module_path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return "misc"


def page_template(
    title: str,
    description: str,
    breadcrumbs: list[tuple[str, str | None]],
    content_html: str,
    prefix: str,
    current_rel: str,
    module_nav: dict[str, list[tuple[str, str]]],
) -> str:
    breadcrumb_html = "".join(
        f"<li>{h(label)}</li>" if href is None else f"<li><a href=\"{h(href)}\">{h(label)}</a></li>"
        for label, href in breadcrumbs
    )

    static_pages = [
        ("Home", "index.html"),
        ("Installation", "installation.html"),
        ("Quick Start", "quickstart.html"),
        ("Architecture", "architecture.html"),
        ("API", "api/index.html"),
        ("Configuration", "configuration.html"),
        ("Usage", "usage.html"),
        ("Testing", "testing.html"),
        ("Contributing", "contributing.html"),
        ("FAQ", "faq.html"),
    ]

    nav_items = []
    for label, rel in static_pages:
        href = prefix + rel
        cls = "nav-link active" if current_rel == rel else "nav-link"
        nav_items.append(f"<li><a class=\"{cls}\" href=\"{h(href)}\">{h(label)}</a></li>")

    modules_html = []
    for category in sorted(module_nav):
        links = []
        for mod_label, mod_rel in sorted(module_nav[category], key=lambda x: x[0]):
            href = prefix + mod_rel
            cls = "api-link active" if current_rel == mod_rel else "api-link"
            links.append(f"<li><a class=\"{cls}\" href=\"{h(href)}\">{h(mod_label)}</a></li>")
        modules_html.append(
            "<details class=\"nav-group\">"
            f"<summary>{h(category)}</summary>"
            "<ul class=\"api-links\">"
            + "".join(links)
            + "</ul></details>"
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta name=\"description\" content=\"{h(description)}\" />
  <title>{h(title)} | SARPyX Documentation</title>
  <link rel=\"stylesheet\" href=\"{h(prefix)}assets/styles.css\" />
  <script defer src=\"{h(prefix)}assets/scripts.js\"></script>
</head>
<body>
  <a class=\"skip-link\" href=\"#main-content\">Skip to content</a>
  <header class=\"topbar\" role=\"banner\">
    <button class=\"sidebar-toggle\" data-sidebar-toggle aria-controls=\"sidebar\" aria-expanded=\"false\">Menu</button>
    <a class=\"brand\" href=\"{h(prefix)}index.html\">SARPyX Docs</a>
    <label class=\"search-wrap\" for=\"site-search\">
      <span class=\"visually-hidden\">Filter this page</span>
      <input id=\"site-search\" type=\"search\" placeholder=\"Filter current page sections\" />
    </label>
  </header>

  <div class=\"layout\">
    <aside id=\"sidebar\" class=\"sidebar\" aria-label=\"Sidebar navigation\">
      <nav>
        <h2 class=\"nav-title\">Documentation</h2>
        <ul class=\"nav-list\">
          {''.join(nav_items)}
        </ul>
        <h2 class=\"nav-title\">API Modules</h2>
        {''.join(modules_html)}
      </nav>
    </aside>

    <main id=\"main-content\" class=\"content\" role=\"main\">
      <nav class=\"breadcrumbs\" aria-label=\"Breadcrumb\">
        <ol>{breadcrumb_html}</ol>
      </nav>
      <article class=\"doc-page\" data-searchable data-search=\"{h(title)}\">
        {content_html}
      </article>
      <footer class=\"page-footer\">
        <p>Generated from repository sources on {h(now_iso)}.</p>
      </footer>
    </main>
  </div>
</body>
</html>
"""


def make_callout(kind: str, text: str) -> str:
    return f"<aside class=\"callout {h(kind)}\"><p>{h(text)}</p></aside>"


def render_parameter_table(raw_params: list[dict[str, Any]], param_desc: dict[str, str]) -> str:
    rows: list[str] = []
    for param in raw_params:
        name = str(param.get("name", ""))
        if name in {"self", "cls"}:
            continue
        lookup_name = name.lstrip("*")
        desc = param_desc.get(name) or param_desc.get(lookup_name) or "inferred from implementation."
        rows.append(
            "<tr>"
            f"<td><code>{h(name)}</code></td>"
            f"<td><code>{h(param.get('annotation') or 'inferred from implementation')}</code></td>"
            f"<td>{'yes' if param.get('required') else 'no'}</td>"
            f"<td><code>{h(format_default(param.get('default')))}</code></td>"
            f"<td>{h(desc)}</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td><code>-</code></td><td><code>-</code></td><td>no</td><td><code>-</code></td>"
            "<td>No explicit public parameters; behavior inferred from implementation.</td></tr>"
        )

    return (
        "<table class=\"api-table\">"
        "<thead><tr><th>Parameter</th><th>Type</th><th>Required</th><th>Default</th><th>Description</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_function_item(module_path: str, func: dict[str, Any]) -> str:
    name = func.get("name", "unknown")
    signature = func.get("signature") or f"{name}(...)"
    doc = func.get("doc")
    raw_params = func.get("raw_parameters", [])
    returns = func.get("returns") or "inferred from implementation"
    exceptions = func.get("exceptions", [])
    side_effects = func.get("side_effects", [])
    line = func.get("line", "?")

    desc = doc.splitlines()[0].strip() if doc else "inferred from implementation."
    param_desc = parse_doc_params(doc)
    returns_desc = parse_doc_section(doc, {"Returns:", "Returns"}) or "inferred from implementation."
    examples_desc = parse_doc_section(doc, {"Examples:", "Examples"})
    if examples_desc:
        example_code = examples_desc
    else:
        example_code = build_example(module_to_import(module_path), name, raw_params)
    edge_cases = infer_edge_cases(raw_params, exceptions, doc)

    anchor = slugify(f"{module_path}-{name}")
    exceptions_html = (
        "<ul>" + "".join(f"<li><code>{h(ex)}</code></li>" for ex in exceptions) + "</ul>"
        if exceptions
        else "<p>None explicitly documented; inferred from implementation.</p>"
    )
    side_effects_html = (
        "<ul>" + "".join(f"<li>{h(item)}</li>" for item in side_effects) + "</ul>"
        if side_effects
        else "<p>None explicitly documented; inferred from implementation.</p>"
    )

    return f"""
<details class=\"api-item\" id=\"{h(anchor)}\" data-searchable data-search=\"{h(name)} {h(signature)}\">
  <summary><code>{h(name)}</code> <span class=\"api-kind\">function</span></summary>
  <div class=\"api-body\">
    <p>{h(desc)}</p>
    <p><strong>File location:</strong> <code>{h(module_path)}:{h(line)}</code></p>
    <h3 id=\"{h(anchor)}-signature\">Signature</h3>
    <pre><code class=\"language-python\">{h(signature)}</code></pre>
    <h3 id=\"{h(anchor)}-parameters\">Parameters</h3>
    {render_parameter_table(raw_params, param_desc)}
    <h3 id=\"{h(anchor)}-returns\">Return Type</h3>
    <p><code>{h(returns)}</code></p>
    <p>{h(returns_desc)}</p>
    <h3 id=\"{h(anchor)}-exceptions\">Exceptions</h3>
    {exceptions_html}
    <h3 id=\"{h(anchor)}-side-effects\">Side Effects</h3>
    {side_effects_html}
    <h3 id=\"{h(anchor)}-example\">Example Usage</h3>
    <pre><code class=\"language-python\">{h(example_code)}</code></pre>
    <h3 id=\"{h(anchor)}-edge-cases\">Edge Cases</h3>
    <p>{h(edge_cases)}</p>
  </div>
</details>
"""


def render_method_item(module_path: str, cls_name: str, method: dict[str, Any]) -> str:
    name = method.get("name", "unknown")
    signature = method.get("signature") or f"{name}(...)"
    doc = method.get("doc")
    raw_params = method.get("raw_parameters", [])
    returns = method.get("returns") or "inferred from implementation"
    exceptions = method.get("exceptions", [])
    side_effects = method.get("side_effects", [])
    line = method.get("line", "?")

    desc = doc.splitlines()[0].strip() if doc else "inferred from implementation."
    param_desc = parse_doc_params(doc)
    returns_desc = parse_doc_section(doc, {"Returns:", "Returns"}) or "inferred from implementation."
    examples_desc = parse_doc_section(doc, {"Examples:", "Examples"})
    if examples_desc:
        example_code = examples_desc
    else:
        example_code = build_example(module_to_import(module_path), name, raw_params, class_name=cls_name)
    edge_cases = infer_edge_cases(raw_params, exceptions, doc)

    anchor = slugify(f"{module_path}-{cls_name}-{name}")
    exceptions_html = (
        "<ul>" + "".join(f"<li><code>{h(ex)}</code></li>" for ex in exceptions) + "</ul>"
        if exceptions
        else "<p>None explicitly documented; inferred from implementation.</p>"
    )
    side_effects_html = (
        "<ul>" + "".join(f"<li>{h(item)}</li>" for item in side_effects) + "</ul>"
        if side_effects
        else "<p>None explicitly documented; inferred from implementation.</p>"
    )

    return f"""
<details class=\"api-item method\" id=\"{h(anchor)}\" data-searchable data-search=\"{h(cls_name)} {h(name)} {h(signature)}\">
  <summary><code>{h(cls_name)}.{h(name)}</code> <span class=\"api-kind\">method</span></summary>
  <div class=\"api-body\">
    <p>{h(desc)}</p>
    <p><strong>File location:</strong> <code>{h(module_path)}:{h(line)}</code></p>
    <h4 id=\"{h(anchor)}-signature\">Signature</h4>
    <pre><code class=\"language-python\">{h(signature)}</code></pre>
    <h4 id=\"{h(anchor)}-parameters\">Parameters</h4>
    {render_parameter_table(raw_params, param_desc)}
    <h4 id=\"{h(anchor)}-returns\">Return Type</h4>
    <p><code>{h(returns)}</code></p>
    <p>{h(returns_desc)}</p>
    <h4 id=\"{h(anchor)}-exceptions\">Exceptions</h4>
    {exceptions_html}
    <h4 id=\"{h(anchor)}-side-effects\">Side Effects</h4>
    {side_effects_html}
    <h4 id=\"{h(anchor)}-example\">Example Usage</h4>
    <pre><code class=\"language-python\">{h(example_code)}</code></pre>
    <h4 id=\"{h(anchor)}-edge-cases\">Edge Cases</h4>
    <p>{h(edge_cases)}</p>
  </div>
</details>
"""


def render_class_item(module_path: str, cls: dict[str, Any]) -> str:
    cls_name = cls.get("name", "UnknownClass")
    signature = cls.get("signature") or f"class {cls_name}"
    doc = cls.get("doc") or "inferred from implementation."
    line = cls.get("line", "?")
    methods = cls.get("methods", [])

    method_html = "".join(render_method_item(module_path, cls_name, m) for m in methods)
    constructor_note = "Constructor parameters are inferred from non-public __init__ implementation."

    anchor = slugify(f"{module_path}-{cls_name}")

    return f"""
<details class=\"api-item class-item\" id=\"{h(anchor)}\" data-searchable data-search=\"{h(cls_name)} {h(signature)}\">
  <summary><code>{h(cls_name)}</code> <span class=\"api-kind\">class</span></summary>
  <div class=\"api-body\">
    <p>{h(doc)}</p>
    <p><strong>File location:</strong> <code>{h(module_path)}:{h(line)}</code></p>
    <h3 id=\"{h(anchor)}-signature\">Class Signature</h3>
    <pre><code class=\"language-python\">{h(signature)}</code></pre>
    <h3 id=\"{h(anchor)}-parameters\">Constructor Parameters</h3>
    {make_callout('info', constructor_note)}
    <h3 id=\"{h(anchor)}-returns\">Return Type</h3>
    <p><code>{h(cls_name)}</code> instances.</p>
    <h3 id=\"{h(anchor)}-exceptions\">Exceptions</h3>
    <p>Construction/runtime exceptions are inferred from implementation and method-level documentation.</p>
    <h3 id=\"{h(anchor)}-side-effects\">Side Effects</h3>
    <p>See method-level side effects below.</p>
    <h3 id=\"{h(anchor)}-example\">Example Usage</h3>
    <pre><code class=\"language-python\">from {h(module_to_import(module_path))} import {h(cls_name)}

obj = {h(cls_name)}(...)  # inferred from implementation
</code></pre>
    <h3 id=\"{h(anchor)}-edge-cases\">Edge Cases</h3>
    <p>No class-level edge-case section is explicitly documented; rely on method-level checks and raised exceptions.</p>
    <h3 id=\"{h(anchor)}-methods\">Public Methods ({len(methods)})</h3>
    {method_html if method_html else '<p>No public methods detected.</p>'}
  </div>
</details>
"""


def module_tree_snippet(module_items: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in module_items:
        parts = item["module"].split("/")
        if len(parts) > 2:
            grouped[parts[1]].append("/".join(parts[2:]))
        else:
            grouped["root"].append(item["module"])

    lines = ["sarpyx/"]
    for category in sorted(grouped):
        lines.append(f"  {category}/")
        for rel in sorted(grouped[category])[:16]:
            lines.append(f"    {rel}")
        if len(grouped[category]) > 16:
            lines.append(f"    ... ({len(grouped[category]) - 16} more files)")
    return "\n".join(lines)


def collect_env_reference() -> list[dict[str, str]]:
    return [
        {
            "name": "gpt_path / GPT_PATH",
            "default": "None",
            "source": "sarpyx/cli/worldsar.py",
            "description": "Path to SNAP GPT executable used by worldsar pipeline.",
            "security": "Prefer absolute trusted path; avoid untrusted binaries in PATH.",
        },
        {
            "name": "grid_path / GRID_PATH",
            "default": "None",
            "source": "sarpyx/cli/worldsar.py, Makefile",
            "description": "Path to GeoJSON tiling grid.",
            "security": "Validate file provenance; malformed geometry can break tiling workflows.",
        },
        {
            "name": "db_dir / DB_DIR",
            "default": "None",
            "source": "sarpyx/cli/worldsar.py",
            "description": "Output directory for tile database artifacts.",
            "security": "Use writable path with least privilege.",
        },
        {
            "name": "cuts_outdir / OUTPUT_CUTS_DIR",
            "default": "None",
            "source": "sarpyx/cli/worldsar.py",
            "description": "Output directory for generated raster tiles.",
            "security": "Avoid writing to shared sensitive directories.",
        },
        {
            "name": "base_path / BASE_PATH",
            "default": "project root",
            "source": "sarpyx/cli/worldsar.py",
            "description": "Base path used by worldsar fallback filesystem operations.",
            "security": "Keep under controlled workspace.",
        },
        {
            "name": "SNAP_USERDIR / snap_userdir",
            "default": "<project_root>/.snap",
            "source": "sarpyx/cli/worldsar.py, sarpyx/snapflow/engine.py",
            "description": "SNAP user configuration and cache directory.",
            "security": "Contains execution metadata; protect in multi-user environments.",
        },
        {
            "name": "orbit_base_url / ORBIT_BASE_URL",
            "default": "https://step.esa.int/auxdata/orbits/Sentinel-1",
            "source": "sarpyx/cli/worldsar.py",
            "description": "Base URL for Sentinel orbit prefetch.",
            "security": "Use trusted HTTPS endpoint only.",
        },
        {
            "name": "HF_TOKEN",
            "default": "unset",
            "source": "sarpyx/cli/upload.py",
            "description": "Hugging Face access token for upload operations.",
            "security": "Treat as secret; do not commit to repository.",
        },
        {
            "name": "JAVA_HOME",
            "default": "/usr/lib/jvm/java-8-openjdk-amd64 (container)",
            "source": "Dockerfile, docker-compose.yml",
            "description": "Java runtime required by SNAP.",
            "security": "Pin to trusted JRE installation.",
        },
        {
            "name": "SNAP_HOME",
            "default": "/snap12 or /workspace/snap12 (container)",
            "source": "Dockerfile, entrypoint.sh",
            "description": "SNAP installation directory.",
            "security": "Read-only in production container images where possible.",
        },
        {
            "name": "SNAP_SKIP_UPDATES",
            "default": "1",
            "source": "entrypoint.sh",
            "description": "If set to 1, startup script skips SNAP online update check.",
            "security": "Disabling auto-updates improves reproducibility.",
        },
        {
            "name": "JUPYTER_ENABLE_LAB / JUPYTER_TOKEN / JUPYTER_ALLOW_INSECURE_WRITES",
            "default": "compose defaults",
            "source": "docker-compose.yml",
            "description": "Jupyter runtime behavior inside containerized workflow.",
            "security": "Setting empty token is insecure outside localhost-bound environments.",
        },
    ]


def figma_specs() -> list[dict[str, Any]]:
    return [
        {
            "diagram_name": "High-level system architecture",
            "purpose": "Show top-level package boundaries and external dependencies.",
            "nodes": [
                {"id": "cli", "label": "CLI Layer", "type": "layer"},
                {"id": "core", "label": "Processing Core", "type": "layer"},
                {"id": "utils", "label": "Utilities", "type": "layer"},
                {"id": "science", "label": "Science Indices", "type": "layer"},
                {"id": "sla", "label": "Sub-look Analysis", "type": "layer"},
                {"id": "snapflow", "label": "SNAP Flow", "type": "integration"},
                {"id": "ext_snap", "label": "ESA SNAP GPT", "type": "external"},
                {"id": "ext_hf", "label": "Hugging Face Hub", "type": "external"},
                {"id": "ext_io", "label": "Filesystem / Zarr", "type": "external"},
            ],
            "edges": [
                {"source": "cli", "target": "core", "label": "decode/focus"},
                {"source": "cli", "target": "snapflow", "label": "shipdet/worldsar"},
                {"source": "core", "target": "utils", "label": "shared helpers"},
                {"source": "science", "target": "utils", "label": "array ops"},
                {"source": "sla", "target": "utils", "label": "metrics/io"},
                {"source": "snapflow", "target": "ext_snap", "label": "subprocess gpt"},
                {"source": "cli", "target": "ext_hf", "label": "upload"},
                {"source": "core", "target": "ext_io", "label": "read/write"},
            ],
            "suggested_layout_strategy": "Layered horizontal swimlanes: interfaces top, internals middle, externals bottom.",
            "suggested_dark_theme_colors": {
                "background": "#0f1117",
                "surface": "#171a23",
                "layer": "#273044",
                "integration": "#2f3a56",
                "external": "#3f4b6a",
                "edge": "#7aa2f7",
                "text": "#e6edf3",
            },
            "component_grouping_guidance": "Group package nodes by top-level namespace; keep external nodes visually separated with dashed borders.",
        },
        {
            "diagram_name": "Module dependency graph",
            "purpose": "Show coupling between high-traffic modules.",
            "nodes": [
                {"id": "cli_main", "label": "sarpyx.cli.main", "type": "module"},
                {"id": "cli_worldsar", "label": "sarpyx.cli.worldsar", "type": "module"},
                {"id": "snap_engine", "label": "sarpyx.snapflow.engine", "type": "module"},
                {"id": "focus_core", "label": "sarpyx.processor.core.focus", "type": "module"},
                {"id": "decode_core", "label": "sarpyx.processor.core.decode", "type": "module"},
                {"id": "zarr_utils", "label": "sarpyx.utils.zarr_utils", "type": "module"},
                {"id": "geos", "label": "sarpyx.utils.geos", "type": "module"},
            ],
            "edges": [
                {"source": "cli_main", "target": "cli_worldsar", "label": "dispatch"},
                {"source": "cli_worldsar", "target": "snap_engine", "label": "GPT wrapper"},
                {"source": "cli_worldsar", "target": "geos", "label": "tiling geometry"},
                {"source": "decode_core", "target": "zarr_utils", "label": "persist decode output"},
                {"source": "focus_core", "target": "zarr_utils", "label": "persist focused slices"},
            ],
            "suggested_layout_strategy": "Force-directed layout with edge labels; place orchestration modules at center.",
            "suggested_dark_theme_colors": {
                "background": "#0f1117",
                "surface": "#171a23",
                "module": "#26324c",
                "edge": "#8fb8ff",
                "text": "#e6edf3",
            },
            "component_grouping_guidance": "Color-code nodes by namespace: cli, processor, snapflow, utils.",
        },
        {
            "diagram_name": "Data flow diagram",
            "purpose": "Describe decode->focus->analysis->export lifecycle.",
            "nodes": [
                {"id": "input", "label": "Input Product (.SAFE/.dat/.zip)", "type": "data"},
                {"id": "decode", "label": "Decode", "type": "process"},
                {"id": "zarr", "label": "Decoded Zarr", "type": "data"},
                {"id": "focus", "label": "Focus (CoarseRDA)", "type": "process"},
                {"id": "focused", "label": "Focused Arrays", "type": "data"},
                {"id": "analysis", "label": "SLA/Science Metrics", "type": "process"},
                {"id": "outputs", "label": "Exports (GeoTIFF/HDF5/Zarr)", "type": "data"},
            ],
            "edges": [
                {"source": "input", "target": "decode", "label": "read"},
                {"source": "decode", "target": "zarr", "label": "write"},
                {"source": "zarr", "target": "focus", "label": "load"},
                {"source": "focus", "target": "focused", "label": "persist"},
                {"source": "focused", "target": "analysis", "label": "compute"},
                {"source": "analysis", "target": "outputs", "label": "export"},
            ],
            "suggested_layout_strategy": "Left-to-right pipeline with process nodes above data nodes.",
            "suggested_dark_theme_colors": {
                "background": "#0f1117",
                "surface": "#171a23",
                "process": "#2f3f5f",
                "data": "#3c4f77",
                "edge": "#86a8ff",
                "text": "#e6edf3",
            },
            "component_grouping_guidance": "Group decode/focus as core processing stage; keep analysis/export as downstream stage.",
        },
        {
            "diagram_name": "Request lifecycle",
            "purpose": "Model CLI request lifecycle including argument parsing, dispatch, processing, and reporting.",
            "nodes": [
                {"id": "argv", "label": "CLI Args", "type": "event"},
                {"id": "parser", "label": "Argument Parser", "type": "process"},
                {"id": "dispatch", "label": "Subcommand Dispatch", "type": "process"},
                {"id": "op", "label": "Operation Execution", "type": "process"},
                {"id": "log", "label": "Logs/Status", "type": "data"},
                {"id": "exit", "label": "Exit Code", "type": "event"},
            ],
            "edges": [
                {"source": "argv", "target": "parser", "label": "parse"},
                {"source": "parser", "target": "dispatch", "label": "validate"},
                {"source": "dispatch", "target": "op", "label": "run handler"},
                {"source": "op", "target": "log", "label": "emit logs"},
                {"source": "op", "target": "exit", "label": "success/failure"},
            ],
            "suggested_layout_strategy": "Sequence-diagram style horizontal stages.",
            "suggested_dark_theme_colors": {
                "background": "#0f1117",
                "surface": "#171a23",
                "event": "#30415f",
                "process": "#2b3955",
                "data": "#40557f",
                "edge": "#90b4ff",
                "text": "#e6edf3",
            },
            "component_grouping_guidance": "Separate validation steps from execution to highlight failure boundaries.",
        },
        {
            "diagram_name": "Deployment architecture",
            "purpose": "Show local and container execution topology for GitHub Pages docs and runtime services.",
            "nodes": [
                {"id": "dev", "label": "Developer Host", "type": "environment"},
                {"id": "venv", "label": ".venv + uv", "type": "runtime"},
                {"id": "docker", "label": "Docker Container", "type": "runtime"},
                {"id": "snap", "label": "SNAP + GPT", "type": "dependency"},
                {"id": "data", "label": "Mounted Data/Output", "type": "storage"},
                {"id": "pages", "label": "GitHub Pages (docs/)", "type": "deployment"},
            ],
            "edges": [
                {"source": "dev", "target": "venv", "label": "local runs"},
                {"source": "dev", "target": "docker", "label": "compose"},
                {"source": "docker", "target": "snap", "label": "processing"},
                {"source": "docker", "target": "data", "label": "read/write volumes"},
                {"source": "dev", "target": "pages", "label": "publish static docs"},
            ],
            "suggested_layout_strategy": "Clustered deployment view with host and container zones.",
            "suggested_dark_theme_colors": {
                "background": "#0f1117",
                "surface": "#171a23",
                "environment": "#2f3d5a",
                "runtime": "#34476a",
                "dependency": "#40557f",
                "storage": "#315173",
                "deployment": "#2f5b73",
                "edge": "#8ab5ff",
                "text": "#e6edf3",
            },
            "component_grouping_guidance": "Use container boundary frames and separate deployment target frame for GitHub Pages.",
        },
    ]


def make_index_page(project: dict[str, Any], highlights: list[str], module_items: list[dict[str, Any]]) -> str:
    deps = project.get("dependencies", [])
    top_deps = ", ".join(deps[:12]) + (", ..." if len(deps) > 12 else "")

    categories: dict[str, int] = defaultdict(int)
    for item in module_items:
        categories[category_for_module(item["module"])] += 1

    category_rows = "".join(
        f"<tr><td><code>{h(cat)}</code></td><td>{count}</td></tr>" for cat, count in sorted(categories.items())
    )

    feature_items = "".join(f"<li>{h(item)}</li>" for item in (highlights or ["Features inferred from repository implementation."]))

    return f"""
<section>
  <h1 id=\"project-overview\">Project Overview</h1>
  <p><strong>{h(project.get('name', 'sarpyx'))}</strong> is a Python SAR processing toolkit with CLI workflows for decoding, focusing, SNAP-based processing, tiling, and scientific metrics.</p>
  <h2 id=\"problem-statement\">Problem Statement</h2>
  <p>The repository addresses repeatable processing of large SAR products across multiple missions while preserving metadata and providing automation-friendly CLI and Python APIs.</p>
  <h2 id=\"target-audience\">Target Audience</h2>
  <ul>
    <li>Remote-sensing engineers building SAR preprocessing pipelines.</li>
    <li>Researchers running sub-look analysis and derived metrics.</li>
    <li>Platform engineers operating SNAP-backed processing in containers.</li>
  </ul>
  <h2 id=\"key-features\">Key Features</h2>
  <ul>{feature_items}</ul>
  <h2 id=\"high-level-architecture-summary\">High-Level Architecture Summary</h2>
  <p>The package is split into `cli`, `processor`, `snapflow`, `sla`, `science`, and `utils` namespaces. The CLI layer orchestrates decode/focus/tiling flows, while core modules process arrays and persist results (for example to Zarr). SNAP integration wraps GPT command execution for mission-specific pipelines.</p>
  <div class=\"card-grid\">
    <section class=\"card\" data-searchable data-search=\"primary language dependencies\">
      <h3 id=\"primary-languages\">Primary Languages</h3>
      <p>Python, Shell, Dockerfile, Markdown.</p>
    </section>
    <section class=\"card\" data-searchable data-search=\"python version runtime\">
      <h3 id=\"runtime\">Runtime</h3>
      <p>Project requires Python <code>{h(project.get('requires-python', '>=3.11'))}</code>.</p>
    </section>
    <section class=\"card\" data-searchable data-search=\"dependencies\">
      <h3 id=\"dependencies\">Core Dependencies</h3>
      <p>{h(top_deps)}</p>
    </section>
  </div>
  <h2 id=\"module-distribution\">Module Distribution</h2>
  <table>
    <thead><tr><th>Namespace</th><th>Module Count</th></tr></thead>
    <tbody>{category_rows}</tbody>
  </table>
  {make_callout('info', 'API counts and signatures are generated directly from docs/api_inventory_detailed.json (source-derived).')}
</section>
"""


def make_installation_page(project: dict[str, Any]) -> str:
    dependencies = project.get("dependencies", [])
    dep_preview = "\n".join(f"- {d}" for d in dependencies[:20])

    return f"""
<section>
  <h1 id=\"installation\">Installation</h1>
  <h2 id=\"prerequisites\">Prerequisites</h2>
  <ul>
    <li>Python <code>{h(project.get('requires-python', '>=3.11'))}</code>.</li>
    <li><code>uv</code> (recommended by repository tooling).</li>
    <li>SNAP + Java when running SNAP-dependent commands (<code>shipdet</code>, <code>worldsar</code>).</li>
    <li>Docker Engine if using container workflows.</li>
  </ul>

  <h2 id=\"setup-steps\">Setup Steps</h2>
  <pre><code class=\"language-bash\"># repository root
uv venv .venv
source .venv/bin/activate
uv sync
</code></pre>

  <h2 id=\"environment-variables\">Environment Variables</h2>
  <p>Set runtime variables as needed (see <a href=\"configuration.html\">Configuration Reference</a>).</p>
  <pre><code class=\"language-bash\">export GRID_PATH=./grid/grid_10km.geojson
export GPT_PATH=/path/to/gpt
export HF_TOKEN=***
</code></pre>

  <h2 id=\"configuration\">Configuration</h2>
  <p>Primary configuration sources are CLI flags, environment variables, and files such as <code>pyproject.toml</code>, <code>docker-compose.yml</code>, and Makefile targets.</p>

  <h2 id=\"build\">Build</h2>
  <pre><code class=\"language-bash\"># local package install
python -m pip install -e .

# container build
make docker-build
</code></pre>

  <h2 id=\"development\">Development</h2>
  <pre><code class=\"language-bash\">source .venv/bin/activate
uv sync
pytest -q
</code></pre>

  <h2 id=\"production\">Production</h2>
  <p>Production workflows are typically containerized, pinning SNAP and Python dependencies through Docker image builds.</p>
  <pre><code class=\"language-bash\">make recreate
</code></pre>

  <h2 id=\"docker\">Docker</h2>
  <pre><code class=\"language-bash\">docker compose version
make check-grid
make recreate
</code></pre>

  <h2 id=\"troubleshooting\">Troubleshooting</h2>
  <ul>
    <li>SNAP missing: verify <code>SNAP_HOME</code> and <code>gpt</code> in <code>PATH</code>.</li>
    <li>Grid errors: set <code>GRID_PATH</code> or provide <code>./grid/grid_10km.geojson</code>.</li>
    <li>Upload auth failures: set <code>HF_TOKEN</code> or run Hugging Face CLI login.</li>
  </ul>

  {make_callout('tip', 'Dependency preview from pyproject.toml:')}
  <pre><code class=\"language-text\">{h(dep_preview)}</code></pre>
</section>
"""


def make_quickstart_page() -> str:
    return """
<section>
  <h1 id=\"quick-start\">Quick Start</h1>
  <h2 id=\"minimal-example\">Minimal Example</h2>
  <pre><code class=\"language-bash\"># 1) decode
sarpyx decode --input /data/product.dat --output /data/decoded

# 2) focus
sarpyx focus --input /data/decoded/product.zarr --output /data/focused
</code></pre>

  <h2 id=\"cli-usage\">CLI Usage</h2>
  <pre><code class=\"language-bash\">sarpyx --help
sarpyx decode --help
sarpyx focus --help
sarpyx shipdet --help
sarpyx worldsar --help
</code></pre>

  <h2 id=\"common-workflow\">Common Workflow</h2>
  <ol>
    <li>Prepare input products and grid files.</li>
    <li>Run decode/focus (or mission pipeline via worldsar).</li>
    <li>Export products and inspect metrics.</li>
    <li>Optionally upload outputs using <code>sarpyx upload</code>.</li>
  </ol>

  <h2 id=\"python-api-minimal\">Python API Minimal</h2>
  <pre><code class=\"language-python\">from sarpyx.utils.zarr_utils import ZarrManager

manager = ZarrManager("/data/focused/product.zarr")
arr = manager.load()
print(arr.shape)
</code></pre>

  <p>Behavior and imports above are inferred from implementation and current public exports.</p>
</section>
"""


def make_architecture_page(module_items: list[dict[str, Any]], specs: list[dict[str, Any]]) -> str:
    tree = module_tree_snippet(module_items)

    responsibilities_rows = []
    by_category: dict[str, list[str]] = defaultdict(list)
    for item in module_items:
        by_category[category_for_module(item["module"])].append(item["module"])

    for cat in sorted(by_category):
        sample = ", ".join(module_to_import(x) for x in sorted(by_category[cat])[:3])
        responsibilities_rows.append(
            f"<tr><td><code>{h(cat)}</code></td><td>{len(by_category[cat])}</td><td>{h(sample)}{(' ...' if len(by_category[cat]) > 3 else '')}</td></tr>"
        )

    spec_blocks = "\n".join(
        f"<h3 id=\"figma-{slugify(spec['diagram_name'])}\">{h(spec['diagram_name'])}</h3>"
        f"<script type=\"application/json\" id=\"figma-spec-{slugify(spec['diagram_name'])}\">{h(json.dumps(spec, indent=2))}</script>"
        for spec in specs
    )

    return f"""
<section>
  <h1 id=\"architecture\">Architecture</h1>
  <h2 id=\"folder-structure\">Folder Structure</h2>
  <pre><code class=\"language-text\">{h(tree)}</code></pre>

  <h2 id=\"core-modules-and-responsibilities\">Core Modules and Responsibilities</h2>
  <table>
    <thead><tr><th>Namespace</th><th>Modules</th><th>Representative Modules</th></tr></thead>
    <tbody>{''.join(responsibilities_rows)}</tbody>
  </table>

  <h2 id=\"data-flow\">Data Flow</h2>
  <ol>
    <li>Input ingestion (raw products, SAFE archives, or Zarr arrays).</li>
    <li>Decode/focus transforms in <code>sarpyx.processor.core</code>.</li>
    <li>Optional SNAP pipeline operations via <code>sarpyx.snapflow.engine.GPT</code>.</li>
    <li>Persistence via Zarr and export utilities.</li>
    <li>Metric/analysis stages in <code>sarpyx.sla</code> and <code>sarpyx.science</code>.</li>
  </ol>

  <h2 id=\"external-integrations\">External Integrations</h2>
  <ul>
    <li>ESA SNAP GPT command-line tooling.</li>
    <li>Hugging Face Hub for upload workflows.</li>
    <li>Numerical and geospatial stack (NumPy, SciPy, rasterio, geopandas, zarr, dask).</li>
  </ul>

  <h2 id=\"design-patterns\">Design Patterns</h2>
  <ul>
    <li>Command-style CLI dispatch in <code>sarpyx.cli.main</code>.</li>
    <li>Wrapper abstraction for external GPT execution in <code>sarpyx.snapflow.engine.GPT</code>.</li>
    <li>Pipeline orchestration helpers in <code>sarpyx.snapflow.snap2stamps</code> with a deprecated compatibility alias at <code>sarpyx.snapflow.snap2stamps_pipelines</code>.</li>
    <li>Manager-style APIs for structured Zarr data access in <code>sarpyx.utils.zarr_utils</code>.</li>
  </ul>

  <h2 id=\"figma-diagram-specifications\">Figma Diagram Specifications</h2>
  <p>The following JSON blocks are export-ready structured specs for recreation in Figma.</p>
  {spec_blocks}
</section>
"""


def make_configuration_page() -> str:
    env_rows = []
    for row in collect_env_reference():
        env_rows.append(
            "<tr>"
            f"<td><code>{h(row['name'])}</code></td>"
            f"<td><code>{h(row['default'])}</code></td>"
            f"<td><code>{h(row['source'])}</code></td>"
            f"<td>{h(row['description'])}</td>"
            f"<td>{h(row['security'])}</td>"
            "</tr>"
        )

    config_files = [
        "pyproject.toml",
        "uv.lock",
        "pdm.lock",
        "docker-compose.yml",
        "Dockerfile",
        "Makefile",
        "entrypoint.sh",
        "support/snap.varfile",
        "conda/recipe/meta.yaml",
    ]

    return f"""
<section>
  <h1 id=\"configuration-reference\">Configuration Reference</h1>
  <h2 id=\"environment-variables\">Environment Variables</h2>
  <table>
    <thead><tr><th>Name</th><th>Default</th><th>Source</th><th>Description</th><th>Security</th></tr></thead>
    <tbody>{''.join(env_rows)}</tbody>
  </table>

  <h2 id=\"config-files\">Config Files</h2>
  <ul>
    {''.join(f'<li><code>{h(p)}</code></li>' for p in config_files)}
  </ul>

  <h2 id=\"defaults\">Defaults</h2>
  <ul>
    <li>Package metadata and dependencies come from <code>pyproject.toml</code>.</li>
    <li>CLI defaults are defined in each parser function (for example <code>sarpyx/cli/main.py</code> and <code>sarpyx/cli/worldsar.py</code>).</li>
    <li>Container defaults are defined in <code>Dockerfile</code> and <code>docker-compose.yml</code>.</li>
  </ul>

  <h2 id=\"security-considerations\">Security Considerations</h2>
  <ul>
    <li>Store <code>HF_TOKEN</code> in environment or secret manager, never in versioned files.</li>
    <li>Do not expose Jupyter with an empty token outside localhost-bound development contexts.</li>
    <li>Pin trusted SNAP/Java binaries and validate filesystem paths passed to CLI commands.</li>
  </ul>
</section>
"""


def make_usage_page() -> str:
    return """
<section>
  <h1 id=\"usage-guides\">Usage Guides</h1>

  <h2 id=\"common-workflows\">Common Workflows</h2>
  <h3 id=\"workflow-l0-to-focused\">L0 to Focused Product</h3>
  <pre><code class=\"language-bash\">sarpyx unzip --input /data/raw.zip --output /data/extracted
sarpyx decode --input /data/extracted/S1A_*.SAFE --output /data/decoded
sarpyx focus --input /data/decoded/product.zarr --output /data/focused
</code></pre>

  <h3 id=\"workflow-worldsar-tiling\">Mission Pipeline and Tiling</h3>
  <pre><code class=\"language-bash\">sarpyx worldsar \
  --input /data/product.SAFE \
  --output /data/preprocessed \
  --cuts-outdir /data/tiles \
  --grid-path /data/grid/grid_10km.geojson
</code></pre>

  <h2 id=\"advanced-usage\">Advanced Usage</h2>
  <ul>
    <li>Use <code>--gpt-memory</code>, <code>--gpt-parallelism</code>, and <code>--gpt-timeout</code> to tune SNAP runtime.</li>
    <li>Use <code>ProductHandler</code> and <code>ZarrManager</code> APIs for selective slicing and export.</li>
    <li>Use mission-specific worldsar pipelines for Sentinel, TerraSAR-X, COSMO, BIOMASS, and NISAR paths (inferred from implementation).</li>
  </ul>

  <h2 id=\"integration-patterns\">Integration Patterns</h2>
  <ul>
    <li>Pipeline orchestration through shell scripts or workflow managers invoking CLI commands.</li>
    <li>Python-first integration importing processing modules directly for notebook or service use.</li>
    <li>Containerized integration via <code>docker compose</code> for reproducible environments.</li>
  </ul>

  <h2 id=\"performance-considerations\">Performance Considerations</h2>
  <ul>
    <li>Enable slicing for large products to reduce memory pressure during focusing.</li>
    <li>Prefer Zarr chunking defaults unless profiling shows bottlenecks.</li>
    <li>Adjust GPT heap and worker count based on available host memory.</li>
    <li>Use parallel tile extraction settings conservatively to avoid I/O saturation.</li>
  </ul>
</section>
"""


def make_testing_page() -> str:
    tests = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))

    return f"""
<section>
  <h1 id=\"testing\">Testing</h1>

  <h2 id=\"running-tests\">Running Tests</h2>
  <pre><code class=\"language-bash\">source .venv/bin/activate
pytest -q

# container verification
make docker-test
</code></pre>

  <h2 id=\"test-structure\">Test Structure</h2>
  <ul>
    {''.join(f'<li><code>tests/{h(t)}</code></li>' for t in tests)}
  </ul>

  <h2 id=\"coverage\">Coverage</h2>
  <p>No repository-level coverage report artifact or threshold configuration was detected. Coverage commands can be run with pytest-cov if installed; this statement is inferred from implementation and project files.</p>

  <h2 id=\"verification-gates\">Verification Gates</h2>
  <ul>
    <li>Import and CLI smoke tests in <code>tests/test_docker.py</code>.</li>
    <li>Processing equivalence and dtype consistency in <code>tests/test_subaperture_dask.py</code>.</li>
    <li>Graph mapping checks in <code>tests/test_snap2stamps_pipelines.py</code>.</li>
  </ul>
</section>
"""


def make_contributing_page() -> str:
    return """
<section>
  <h1 id=\"contributing\">Contributing</h1>

  <h2 id=\"setup\">Setup</h2>
  <pre><code class=\"language-bash\">git clone https://github.com/ESA-sarpyx/sarpyx.git
cd sarpyx
uv venv .venv
source .venv/bin/activate
uv sync
pytest -q
</code></pre>

  <h2 id=\"code-standards\">Code Standards</h2>
  <ul>
    <li>Use type annotations and docstrings for public APIs.</li>
    <li>Keep API changes backward-compatible when possible.</li>
    <li>Update tests and docs with behavior changes.</li>
  </ul>
  <p>Automated lint configuration files were not detected in this repository snapshot; standards above are inferred from existing code style and contributor docs.</p>

  <h2 id=\"pr-workflow\">PR Workflow</h2>
  <ol>
    <li>Create a branch from <code>main</code>.</li>
    <li>Implement scoped changes with tests.</li>
    <li>Run local verification (<code>pytest</code> and relevant CLI checks).</li>
    <li>Submit a pull request with clear change summary and evidence.</li>
  </ol>

  <h2 id=\"branching-model\">Branching Model</h2>
  <p>No formal branching policy file was detected; a trunk-based flow from <code>main</code> is inferred from repository structure and contributor guide text.</p>
</section>
"""


def make_faq_page() -> str:
    return """
<section>
  <h1 id=\"faq\">FAQ / Troubleshooting</h1>

  <h2 id=\"known-limitations\">Known Limitations</h2>
  <ul>
    <li>Several TODO markers in worldsar indicate pending support improvements (metadata reorganization, extended modality support).</li>
    <li>Some APIs expose behavior primarily through implementation without full docstring detail.</li>
    <li>SNAP-based flows depend on external installation/runtime state.</li>
  </ul>

  <h2 id=\"frequent-issues\">Frequent Issues</h2>
  <h3 id=\"issue-gpt-not-found\">GPT not found</h3>
  <p>Set <code>--gpt-path</code> or ensure <code>gpt</code> is on <code>PATH</code> and SNAP is installed.</p>

  <h3 id=\"issue-grid-missing\">Grid file missing</h3>
  <p>Set <code>GRID_PATH</code> or provide <code>./grid/grid_10km.geojson</code> before running compose targets.</p>

  <h3 id=\"issue-upload-auth\">Upload authentication failure</h3>
  <p>Set <code>HF_TOKEN</code> or run Hugging Face CLI login before <code>sarpyx upload</code>.</p>

  <h2 id=\"fixes\">Fixes</h2>
  <pre><code class=\"language-bash\"># SNAP / GPT
which gpt

# Grid precheck
make check-grid

# Container diagnostics
make logs

# CLI help sanity
sarpyx --help
</code></pre>
</section>
"""


def write_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    css = """
:root {
  --bg-primary: #0f1117;
  --bg-surface: #171a23;
  --text-primary: #e6edf3;
  --text-muted: #a9b7c6;
  --accent: #6ea8fe;
  --border-subtle: #2b3242;
  --danger: #ff6b6b;
  --warning: #f7b955;
  --info: #69b8ff;
  --tip: #63d494;
}

* {
  box-sizing: border-box;
}

html, body {
  margin: 0;
  padding: 0;
  min-height: 100%;
  background: radial-gradient(circle at top right, #1a2130 0%, var(--bg-primary) 36%), var(--bg-primary);
  color: var(--text-primary);
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  line-height: 1.55;
}

a {
  color: var(--accent);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

code, pre {
  font-family: "JetBrains Mono", "Fira Code", "Source Code Pro", monospace;
}

pre {
  background: #111623;
  border: 1px solid var(--border-subtle);
  border-radius: 0.625rem;
  padding: 0.9rem;
  overflow-x: auto;
  position: relative;
}

pre code {
  color: #d8e2f2;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
}

th, td {
  border: 1px solid var(--border-subtle);
  padding: 0.55rem 0.7rem;
  vertical-align: top;
}

th {
  background: #1f2635;
  color: var(--text-primary);
  text-align: left;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 1rem;
  border-bottom: 1px solid var(--border-subtle);
  background: rgba(15, 17, 23, 0.9);
  backdrop-filter: blur(8px);
}

.brand {
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: 0.02em;
}

.search-wrap {
  margin-left: auto;
  width: min(440px, 50vw);
}

#site-search {
  width: 100%;
  border-radius: 0.5rem;
  border: 1px solid var(--border-subtle);
  background: #111623;
  color: var(--text-primary);
  padding: 0.5rem 0.7rem;
}

.layout {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
}

.sidebar {
  position: sticky;
  top: 3.1rem;
  height: calc(100vh - 3.1rem);
  overflow-y: auto;
  border-right: 1px solid var(--border-subtle);
  background: rgba(19, 23, 31, 0.92);
  padding: 1rem;
}

.nav-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin: 1rem 0 0.5rem;
}

.nav-list,
.api-links {
  list-style: none;
  margin: 0;
  padding: 0;
}

.nav-link,
.api-link {
  display: block;
  padding: 0.35rem 0.45rem;
  border-radius: 0.4rem;
  color: var(--text-muted);
  font-size: 0.93rem;
}

.nav-link:hover,
.api-link:hover,
.nav-link.active,
.api-link.active {
  color: var(--text-primary);
  background: #252d3d;
  text-decoration: none;
}

.nav-group {
  margin: 0.25rem 0;
}

.nav-group > summary {
  cursor: pointer;
  color: var(--text-muted);
  font-size: 0.92rem;
  padding: 0.3rem 0.1rem;
}

.content {
  padding: 1.2rem 2rem 2rem;
  width: min(1260px, 100%);
}

.breadcrumbs ol {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  color: var(--text-muted);
  font-size: 0.9rem;
}

.breadcrumbs li + li::before {
  content: "/";
  margin-right: 0.4rem;
  color: #6f7b8f;
}

.doc-page h1,
.doc-page h2,
.doc-page h3,
.doc-page h4 {
  scroll-margin-top: 5rem;
}

.doc-page h1 {
  margin-top: 0.6rem;
  font-size: clamp(1.6rem, 2.4vw, 2.1rem);
}

.doc-page h2 {
  margin-top: 1.5rem;
  font-size: 1.35rem;
}

.doc-page h3 {
  margin-top: 1.1rem;
  font-size: 1.1rem;
}

.doc-page p,
.doc-page li {
  color: #d5deea;
}

.doc-page ul,
.doc-page ol {
  padding-left: 1.1rem;
}

.anchor-link {
  margin-left: 0.45rem;
  color: #7f8ca1;
  font-size: 0.88em;
  opacity: 0;
}

h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link {
  opacity: 1;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.9rem;
  margin-top: 0.8rem;
}

.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 0.7rem;
  padding: 0.85rem;
}

.api-item {
  border: 1px solid var(--border-subtle);
  border-radius: 0.7rem;
  margin: 0.8rem 0;
  background: #141925;
}

.api-item > summary {
  list-style: none;
  cursor: pointer;
  padding: 0.7rem 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.api-item > summary::-webkit-details-marker {
  display: none;
}

.api-body {
  padding: 0 0.85rem 0.85rem;
  border-top: 1px solid var(--border-subtle);
}

.api-kind {
  margin-left: auto;
  font-size: 0.75rem;
  text-transform: uppercase;
  color: #b8c3d9;
  border: 1px solid #334058;
  border-radius: 0.3rem;
  padding: 0.1rem 0.35rem;
}

.callout {
  border-left: 4px solid var(--info);
  background: #172131;
  border-radius: 0.5rem;
  padding: 0.65rem 0.85rem;
  margin: 0.9rem 0;
}

.callout.info { border-left-color: var(--info); }
.callout.warning { border-left-color: var(--warning); }
.callout.tip { border-left-color: var(--tip); }
.callout.danger { border-left-color: var(--danger); }

.copy-button {
  position: absolute;
  top: 0.45rem;
  right: 0.45rem;
  border: 1px solid var(--border-subtle);
  background: #1a2232;
  color: var(--text-primary);
  border-radius: 0.35rem;
  padding: 0.2rem 0.45rem;
  font-size: 0.75rem;
  cursor: pointer;
}

.copy-button:hover {
  background: #243048;
}

.sidebar-toggle {
  display: none;
  border: 1px solid var(--border-subtle);
  background: #1a2232;
  color: var(--text-primary);
  border-radius: 0.45rem;
  padding: 0.3rem 0.55rem;
}

.page-footer {
  margin-top: 1.5rem;
  border-top: 1px solid var(--border-subtle);
  padding-top: 1rem;
  color: var(--text-muted);
  font-size: 0.88rem;
}

.skip-link {
  position: absolute;
  left: -999px;
  top: auto;
}

.skip-link:focus {
  left: 0.8rem;
  top: 0.5rem;
  z-index: 99;
  background: #243048;
  color: var(--text-primary);
  padding: 0.45rem;
  border-radius: 0.3rem;
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.hidden-by-filter {
  display: none !important;
}

@media (max-width: 1024px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .sidebar {
    position: fixed;
    top: 3.2rem;
    left: 0;
    width: min(86vw, 340px);
    transform: translateX(-102%);
    transition: transform 0.2s ease-in-out;
    z-index: 30;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
  }

  body.sidebar-open .sidebar {
    transform: translateX(0);
  }

  .sidebar-toggle {
    display: inline-block;
  }

  .search-wrap {
    width: min(50vw, 300px);
  }

  .content {
    padding: 1rem;
  }
}
""".strip()

    js = """
(() => {
  const sidebarButton = document.querySelector('[data-sidebar-toggle]');
  if (sidebarButton) {
    sidebarButton.addEventListener('click', () => {
      const opened = document.body.classList.toggle('sidebar-open');
      sidebarButton.setAttribute('aria-expanded', String(opened));
    });
  }

  // Add anchor links to headings.
  const headingSelector = 'article h1, article h2, article h3, article h4';
  for (const heading of document.querySelectorAll(headingSelector)) {
    if (!heading.id) {
      const slug = heading.textContent
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
      if (slug) heading.id = slug;
    }
    if (!heading.id) continue;
    if (heading.querySelector('.anchor-link')) continue;
    const anchor = document.createElement('a');
    anchor.className = 'anchor-link';
    anchor.href = `#${heading.id}`;
    anchor.setAttribute('aria-label', `Link to ${heading.textContent.trim()}`);
    anchor.textContent = '#';
    heading.appendChild(anchor);
  }

  // Add copy buttons to code blocks.
  for (const pre of document.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    if (!code) continue;
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || '');
        button.textContent = 'Copied';
        setTimeout(() => (button.textContent = 'Copy'), 1200);
      } catch (_err) {
        button.textContent = 'Failed';
        setTimeout(() => (button.textContent = 'Copy'), 1200);
      }
    });
    pre.appendChild(button);
  }

  // Simple in-page filter for search-ready sections.
  const searchInput = document.getElementById('site-search');
  if (searchInput) {
    const candidates = Array.from(document.querySelectorAll('[data-searchable]'));
    const runFilter = () => {
      const q = searchInput.value.trim().toLowerCase();
      for (const node of candidates) {
        if (!q) {
          node.classList.remove('hidden-by-filter');
          continue;
        }
        const haystack = [node.getAttribute('data-search') || '', node.textContent || '']
          .join(' ')
          .toLowerCase();
        if (haystack.includes(q)) {
          node.classList.remove('hidden-by-filter');
        } else {
          node.classList.add('hidden-by-filter');
        }
      }
    };
    searchInput.addEventListener('input', runFilter);
  }
})();
""".strip()

    (ASSETS_DIR / "styles.css").write_text(css + "\n", encoding="utf-8")
    (ASSETS_DIR / "scripts.js").write_text(js + "\n", encoding="utf-8")


def generate_api_pages(module_items: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, Any]]:
    module_nav: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # Precompute complete module navigation so every generated page gets the full sidebar.
    for item in module_items:
        module_path = item["module"]
        category = category_for_module(module_path)
        rel = f"api/{module_slug(module_path)}.html"
        module_nav[category].append((module_to_import(module_path), rel))

    coverage = {
        "modules": 0,
        "functions": 0,
        "classes": 0,
        "methods": 0,
        "generated_module_pages": 0,
        "module_page_files": [],
        "missing_modules": [],
    }

    module_index_rows = []

    for item in sorted(module_items, key=lambda x: x["module"]):
        module_path = item["module"]
        import_path = module_to_import(module_path)
        slug = module_slug(module_path)
        rel = f"api/{slug}.html"
        out_path = DOCS_DIR / rel

        public_functions = item.get("public_functions", [])
        public_classes = item.get("public_classes", [])
        all_exports = item.get("all") or []

        coverage["modules"] += 1
        coverage["functions"] += len(public_functions)
        coverage["classes"] += len(public_classes)
        coverage["methods"] += sum(len(c.get("methods", [])) for c in public_classes)

        module_desc = item.get("description") or "No module docstring available; module purpose is inferred from implementation."

        exports_html = (
            "<ul>" + "".join(f"<li><code>{h(name)}</code></li>" for name in all_exports) + "</ul>"
            if all_exports
            else "<p>No explicit <code>__all__</code> list. Public symbols inferred from implementation.</p>"
        )

        function_html = "".join(render_function_item(module_path, fn) for fn in public_functions)
        class_html = "".join(render_class_item(module_path, cls) for cls in public_classes)

        content = f"""
<section>
  <h1 id=\"module-{h(slug)}\">Module: <code>{h(import_path)}</code></h1>
  <p><strong>File:</strong> <code>{h(module_path)}</code></p>
  <p>{h(module_desc)}</p>

  <h2 id=\"module-exports\">Exported Symbols (<code>__all__</code>)</h2>
  {exports_html}

  <h2 id=\"public-functions\">Public Functions ({len(public_functions)})</h2>
  {function_html if function_html else '<p>No public top-level functions detected.</p>'}

  <h2 id=\"public-classes\">Public Classes ({len(public_classes)})</h2>
  {class_html if class_html else '<p>No public classes detected.</p>'}
</section>
"""

        page = page_template(
            title=f"API: {import_path}",
            description=f"Exhaustive API reference for {import_path}",
            breadcrumbs=[("Home", "../index.html"), ("API", "index.html"), (import_path, None)],
            content_html=content,
            prefix="../",
            current_rel=rel,
            module_nav=module_nav,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(page, encoding="utf-8")

        coverage["generated_module_pages"] += 1
        coverage["module_page_files"].append(rel)

        module_index_rows.append(
            "<tr>"
            f"<td><a href=\"{h(slug)}.html\"><code>{h(import_path)}</code></a></td>"
            f"<td>{len(public_functions)}</td>"
            f"<td>{len(public_classes)}</td>"
            f"<td>{sum(len(c.get('methods', [])) for c in public_classes)}</td>"
            f"<td><code>{h(module_path)}</code></td>"
            "</tr>"
        )

    api_index_content = f"""
<section>
  <h1 id=\"api-reference\">API Reference</h1>
  <p>This section is generated from source-derived inventory and covers all detected public/exported modules, functions, classes, and methods.</p>

  <div class=\"card-grid\">
    <section class=\"card\"><h2 id=\"api-total-modules\">Modules</h2><p>{coverage['modules']}</p></section>
    <section class=\"card\"><h2 id=\"api-total-functions\">Functions</h2><p>{coverage['functions']}</p></section>
    <section class=\"card\"><h2 id=\"api-total-classes\">Classes</h2><p>{coverage['classes']}</p></section>
    <section class=\"card\"><h2 id=\"api-total-methods\">Methods</h2><p>{coverage['methods']}</p></section>
  </div>

  <h2 id=\"api-module-index\">Module Index</h2>
  <table>
    <thead><tr><th>Module</th><th>Functions</th><th>Classes</th><th>Methods</th><th>File</th></tr></thead>
    <tbody>{''.join(module_index_rows)}</tbody>
  </table>

  {make_callout('info', 'Entries with missing prose are explicitly marked as inferred from implementation.')}
</section>
"""

    api_index_page = page_template(
        title="API Reference",
        description="Exhaustive public API inventory for sarpyx",
        breadcrumbs=[("Home", "../index.html"), ("API", None)],
        content_html=api_index_content,
        prefix="../",
        current_rel="api/index.html",
        module_nav=module_nav,
    )
    (API_DIR / "index.html").write_text(api_index_page, encoding="utf-8")

    return module_nav, coverage


def generate_static_pages(project: dict[str, Any], highlights: list[str], module_items: list[dict[str, Any]], module_nav: dict[str, list[tuple[str, str]],], specs: list[dict[str, Any]]) -> None:
    pages: list[tuple[str, str, str]] = [
        (
            "index.html",
            "SARPyX Documentation",
            make_index_page(project, highlights, module_items),
        ),
        (
            "installation.html",
            "Installation",
            make_installation_page(project),
        ),
        (
            "quickstart.html",
            "Quick Start",
            make_quickstart_page(),
        ),
        (
            "architecture.html",
            "Architecture",
            make_architecture_page(module_items, specs),
        ),
        (
            "configuration.html",
            "Configuration Reference",
            make_configuration_page(),
        ),
        (
            "usage.html",
            "Usage Guides",
            make_usage_page(),
        ),
        (
            "testing.html",
            "Testing",
            make_testing_page(),
        ),
        (
            "contributing.html",
            "Contributing",
            make_contributing_page(),
        ),
        (
            "faq.html",
            "FAQ / Troubleshooting",
            make_faq_page(),
        ),
    ]

    for rel, title, content in pages:
        breadcrumbs = [("Home", None)] if rel == "index.html" else [("Home", "index.html"), (title, None)]
        page = page_template(
            title=title,
            description=title,
            breadcrumbs=breadcrumbs,
            content_html=content,
            prefix="",
            current_rel=rel,
            module_nav=module_nav,
        )
        (DOCS_DIR / rel).write_text(page, encoding="utf-8")


def build_tree_listing() -> list[str]:
    roots = [
        "docs/index.html",
        "docs/installation.html",
        "docs/quickstart.html",
        "docs/architecture.html",
        "docs/configuration.html",
        "docs/usage.html",
        "docs/testing.html",
        "docs/contributing.html",
        "docs/faq.html",
        "docs/assets/styles.css",
        "docs/assets/scripts.js",
        "docs/api/index.html",
    ]
    api_pages = sorted(str(p.relative_to(ROOT)) for p in DOCS_DIR.glob("api/sarpyx_*.html"))
    return roots + api_pages


def main() -> None:
    module_items = collect_inventory()
    project = parse_pyproject()
    highlights = parse_readme_highlights()
    specs = figma_specs()

    write_assets()
    module_nav, coverage = generate_api_pages(module_items)
    generate_static_pages(project, highlights, module_items, module_nav, specs)

    # Write Figma specs as standalone JSON artifact too.
    (DOCS_DIR / "figma_diagram_specs.json").write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")

    inventory_module_paths = sorted(item["module"] for item in module_items)
    generated_module_paths = sorted(module_to_import(item["module"]) for item in module_items)
    coverage_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_modules": len(module_items),
        "inventory_functions": coverage["functions"],
        "inventory_classes": coverage["classes"],
        "inventory_methods": coverage["methods"],
        "generated_module_pages": coverage["generated_module_pages"],
        "module_page_files": coverage["module_page_files"],
        "missing_modules": coverage["missing_modules"],
        "inventory_module_paths": inventory_module_paths,
        "generated_module_import_paths": generated_module_paths,
        "all_modules_covered": len(module_items) == coverage["generated_module_pages"],
    }
    (DOCS_DIR / "api_coverage_checklist.json").write_text(json.dumps(coverage_report, indent=2) + "\n", encoding="utf-8")

    tree = build_tree_listing()
    (DOCS_DIR / "site_file_tree.txt").write_text("\n".join(tree) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
