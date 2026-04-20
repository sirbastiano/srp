#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "snapflow_v2.ipynb"
OUTPUT_DIRS = [
    ROOT / "docs" / "results" / "snapflow_v2",
    ROOT / "docs" / "_site" / "results" / "snapflow_v2",
]


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cells_by_heading(nb: dict[str, Any], heading: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = nb["cells"]
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and "".join(cell.get("source", [])).strip().startswith(heading):
            for follow in cells[idx + 1 :]:
                if follow.get("cell_type") == "code":
                    return cell, follow
    raise KeyError(f"Heading not found: {heading}")


def first_output_html(cell: dict[str, Any], index: int = 0) -> str:
    outputs = cell.get("outputs", [])
    data = outputs[index]["data"]
    return "".join(data["text/html"])


def first_output_png(cell: dict[str, Any], index: int = 0) -> str:
    outputs = cell.get("outputs", [])
    return outputs[index]["data"]["image/png"]


def stream_text(cell: dict[str, Any]) -> str:
    outputs = cell.get("outputs", [])
    if not outputs:
        return ""
    return "".join(outputs[0].get("text", []))


def parse_stream_json(cell: dict[str, Any]) -> dict[str, Any]:
    return json.loads(stream_text(cell))


def find_map_html(nb: dict[str, Any]) -> str:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            html_items = data.get("text/html")
            if not html_items:
                continue
            joined = "".join(html_items)
            if "folium-map" in joined or "iframe srcdoc=" in joined:
                return joined
    raise KeyError("Folium map HTML not found in notebook outputs")


def sanitize_fragment(text: str) -> str:
    text = text.replace(str(ROOT), "")
    text = re.sub(r"/shared/home/[^\"'\\s<]+", "[redacted]", text)
    return text


def build_kv_grid(title: str, payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        value_html = h(value)
        rows.append(f"<div class='kv-card'><div class='kv-key'>{h(key)}</div><div class='kv-value'>{value_html}</div></div>")
    return f"""
    <section class="panel">
      <h2>{h(title)}</h2>
      <div class="kv-grid">
        {''.join(rows)}
      </div>
    </section>
    """


def image_panel(title: str, subtitle: str, b64: str, anchor: str) -> str:
    src = f"data:image/png;base64,{b64}"
    return f"""
    <section class="panel" id="{h(anchor)}">
      <div class="section-head">
        <h2>{h(title)}</h2>
        <a class="expand-link" href="{src}" target="_blank" rel="noopener">Open full resolution</a>
      </div>
      <p class="section-subtitle">{h(subtitle)}</p>
      <button class="image-button" type="button" data-image-src="{src}" data-image-title="{h(title)}">
        <img src="{src}" alt="{h(title)}" loading="lazy" />
      </button>
    </section>
    """


def build_page(nb: dict[str, Any]) -> str:
    config = parse_stream_json(nb["cells"][3])
    env_info = parse_stream_json(nb["cells"][1])
    burst_info = parse_stream_json(nb["cells"][10])
    process_info = parse_stream_json(nb["cells"][14])

    _, pair_cell = cells_by_heading(nb, "## Resolve the Burst Pair")
    pair_html = sanitize_fragment(first_output_html(pair_cell))

    _, summary_cell = cells_by_heading(nb, "### Output Summary")
    summary_html = sanitize_fragment(first_output_html(summary_cell))

    _, stats_cell = cells_by_heading(nb, "### Raster Statistics")
    coherence_stats_html = sanitize_fragment(first_output_html(stats_cell, 0))
    support_stats_html = sanitize_fragment(first_output_html(stats_cell, 1))

    _, phase_cell = cells_by_heading(nb, "### Wrapped Phase")
    _, coherence_cell = cells_by_heading(nb, "### Coherence")
    _, support_cell = cells_by_heading(nb, "### Terrain Support Raster")
    _, hist_cell = cells_by_heading(nb, "### Coherence Histogram")

    map_html = sanitize_fragment(find_map_html(nb))

    config_subset = {
        "aoi_wkt": config["search"]["aoi_wkt"],
        "start_date": config["search"]["start_date"],
        "end_date": config["search"]["end_date"],
        "polarisation": config["search"]["polarisation"],
        "relative_orbit": 117,
    }
    env_subset = {
        "workspace": env_info["workspace"],
        "gpt_exists": env_info["gpt_exists"],
        "output_dir": env_info["output_dir"],
        "process_dir": env_info["process_dir"],
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="Static SNAPflow v2 InSAR results report" />
  <title>SNAPflow v2 Results</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f1b2d;
      --panel-soft: #15243a;
      --border: #253650;
      --text: #ebf1f8;
      --muted: #9fb0c5;
      --accent: #4fd1c5;
      --accent-2: #8ec5ff;
      --shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
      --radius: 18px;
      --content: 1240px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(79, 209, 197, 0.12), transparent 28rem),
        radial-gradient(circle at top right, rgba(142, 197, 255, 0.1), transparent 24rem),
        var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    a {{ color: var(--accent-2); }}
    .shell {{
      width: min(var(--content), calc(100% - 2rem));
      margin: 0 auto;
      padding: 1.25rem 0 4rem;
    }}
    .toplink {{
      display: inline-flex;
      gap: 0.5rem;
      align-items: center;
      color: var(--muted);
      text-decoration: none;
      margin-bottom: 1.2rem;
    }}
    .hero {{
      background: linear-gradient(180deg, rgba(21, 36, 58, 0.95), rgba(12, 23, 37, 0.95));
      border: 1px solid var(--border);
      border-radius: calc(var(--radius) + 4px);
      padding: 1.75rem;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 0.75rem;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.7rem;
      margin-top: 1rem;
    }}
    .chip {{
      padding: 0.45rem 0.8rem;
      border-radius: 999px;
      background: rgba(79, 209, 197, 0.12);
      border: 1px solid rgba(79, 209, 197, 0.22);
      color: var(--text);
      font-size: 0.9rem;
    }}
    .anchor-nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin-top: 1rem;
      padding: 0.9rem 1rem;
      background: rgba(8, 17, 31, 0.88);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .anchor-nav a {{
      text-decoration: none;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .anchor-nav a:hover {{ color: var(--text); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }}
    .panel {{
      margin-top: 1rem;
      background: rgba(15, 27, 45, 0.92);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 0.75rem;
      font-size: 1.35rem;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
    }}
    .section-subtitle {{
      margin: 0.1rem 0 1rem;
      color: var(--muted);
    }}
    .kv-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.8rem;
    }}
    .kv-card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 0.9rem;
      background: var(--panel-soft);
    }}
    .kv-key {{
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }}
    .kv-value {{
      font-size: 0.98rem;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.02);
    }}
    th, td {{
      padding: 0.75rem 0.8rem;
      border: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(142, 197, 255, 0.08);
      color: var(--text);
    }}
    td {{
      color: #d9e4f0;
    }}
    .image-button {{
      display: block;
      width: 100%;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: zoom-in;
    }}
    .image-button img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }}
    .map-wrap {{
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: white;
      min-height: 520px;
    }}
    .map-wrap iframe {{
      width: 100%;
      min-height: 520px;
      border: 0;
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      background: rgba(3, 9, 18, 0.9);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      z-index: 50;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox figure {{
      margin: 0;
      width: min(1400px, 96vw);
    }}
    .lightbox img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      box-shadow: var(--shadow);
    }}
    .lightbox figcaption {{
      margin-top: 0.75rem;
      color: var(--muted);
      text-align: center;
    }}
    .expand-link {{
      white-space: nowrap;
      font-size: 0.92rem;
    }}
    @media (max-width: 720px) {{
      .shell {{ width: min(var(--content), calc(100% - 1rem)); }}
      .hero, .panel {{ padding: 1rem; }}
      .section-head {{ flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <a class="toplink" href="../../index.html">&larr; Back to documentation</a>
    <section class="hero">
      <h1>SNAPflow v2 Results</h1>
      <p>Static, shareable review page for the latest burst-based InSAR run. The content below is sourced from the executed notebook outputs, keeping the visual diagnostics, tabular summaries, and map in one navigable page.</p>
      <div class="chip-row">
        <span class="chip">Latest-only results</span>
        <span class="chip">Static HTML</span>
        <span class="chip">Notebook-derived</span>
        <span class="chip">Paths redacted</span>
      </div>
    </section>

    <nav class="anchor-nav" aria-label="Section navigation">
      <a href="#run-context">Run Context</a>
      <a href="#pair-selection">Burst Pair</a>
      <a href="#output-summary">Output Summary</a>
      <a href="#raster-stats">Raster Statistics</a>
      <a href="#wrapped-phase">Wrapped Phase</a>
      <a href="#coherence">Coherence</a>
      <a href="#terrain-support">Terrain Support</a>
      <a href="#coherence-hist">Histogram</a>
      <a href="#burst-map">Burst Map</a>
    </nav>

    <div class="grid">
      {build_kv_grid("Run Context", config_subset).replace("<section class=\"panel\">", "<section class=\"panel\" id=\"run-context\">", 1)}
      {build_kv_grid("Environment Snapshot", env_subset)}
    </div>

    <section class="panel" id="pair-selection">
      <h2>Selected Burst Pair</h2>
      <p class="section-subtitle">Master and slave bursts chosen by the notebook for the current interferometric pair.</p>
      {pair_html}
    </section>

    <section class="panel" id="output-summary">
      <h2>Output Summary</h2>
      <p class="section-subtitle">Processing stages and available raster bands exposed by the generated products.</p>
      {sanitize_fragment(summary_html)}
    </section>

    <section class="panel" id="raster-stats">
      <div class="section-head">
        <h2>Raster Statistics</h2>
      </div>
      <p class="section-subtitle">Descriptive statistics derived from the coherence raster and the selected terrain-support raster.</p>
      <div class="grid">
        <section>{sanitize_fragment(coherence_stats_html)}</section>
        <section>{sanitize_fragment(support_stats_html)}</section>
      </div>
    </section>

    {image_panel("Wrapped Phase", "Wrapped phase visualized independently for clearer fringe inspection.", first_output_png(phase_cell), "wrapped-phase")}
    {image_panel("Coherence", "Monotonic colormap used for easier quality interpretation across the scene.", first_output_png(coherence_cell), "coherence")}
    {image_panel("Terrain Support Raster", "Auxiliary terrain-support layer used to contextualize interferometric outputs.", first_output_png(support_cell), "terrain-support")}
    {image_panel("Coherence Histogram", "Distribution view showing central tendency and percentile range.", first_output_png(hist_cell), "coherence-hist")}

    <section class="panel" id="burst-map">
      <h2>Burst Map</h2>
      <p class="section-subtitle">Interactive footprint map extracted from the notebook output and embedded at the bottom of the report.</p>
      <div class="map-wrap">
        {map_html}
      </div>
    </section>
  </div>

  <div class="lightbox" id="lightbox" aria-hidden="true">
    <figure>
      <img id="lightbox-image" alt="" />
      <figcaption id="lightbox-caption"></figcaption>
    </figure>
  </div>

  <script>
    const lightbox = document.getElementById("lightbox");
    const lightboxImage = document.getElementById("lightbox-image");
    const lightboxCaption = document.getElementById("lightbox-caption");
    document.querySelectorAll("[data-image-src]").forEach((button) => {{
      button.addEventListener("click", () => {{
        lightboxImage.src = button.dataset.imageSrc;
        lightboxCaption.textContent = button.dataset.imageTitle || "";
        lightbox.classList.add("open");
        lightbox.setAttribute("aria-hidden", "false");
      }});
    }});
    lightbox.addEventListener("click", () => {{
      lightbox.classList.remove("open");
      lightbox.setAttribute("aria-hidden", "true");
      lightboxImage.src = "";
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    nb = read_notebook()
    page = build_page(nb)
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(page, encoding="utf-8")
        print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
