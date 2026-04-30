---
title: 'sarpyx: Reproducible SAR processing workflows and sub-aperture analysis with Python and ESA SNAP'
tags:
  - Python
  - synthetic aperture radar
  - remote sensing
  - Sentinel-1
  - ESA SNAP
  - SAR processing
authors:
  - name: Roberto Del Prete
    corresponding: true
    affiliation: "1"
  - name: Gabriele Daga
    affiliation: "1"
  - name: Sebastian Fieldhouse
    affiliation: "1"
  - name: Juanfrancisco Amieva
    affiliation: "1"
  - name: Cedric Leonard
    affiliation: "1"
  - name: Valerio Marsocci
    affiliation: "1"
  - name: Eva Gmelich Mejling
    affiliation: "1"
affiliations:
  - index: 1
    name: ESA Φ-lab, European Space Agency, Italy
    ror: "03h3jqn23"
date: 30 April 2026
bibliography: paper.bib
---

# Summary

Synthetic aperture radar (SAR) products are central to Earth observation, but routine
processing still tends to be split across mission-specific utilities, SNAP graphs, shell
scripts, and ad hoc notebook code. `sarpyx` is a Python toolkit that packages this work
into reproducible command-line and Python workflows. It supports decoding and focusing
steps, SNAP-backed preprocessing, tiled exports for large scenes, and sub-aperture
decomposition for analysis of directional content and motion sensitivity. The toolkit is
designed for researchers and engineers who need to move from raw or higher-level SAR
products to analysis-ready outputs without rewriting orchestration code for each mission
or platform.

`sarpyx` combines a CLI for operational runs with importable Python modules for notebooks
and services. The implementation is oriented toward repeatability: environment-based
configuration, container-friendly execution, chunked array storage, and modular
interfaces for orchestration, processing algorithms, scientific metrics, and utility
layers. This makes the same codebase suitable for local prototyping, reproducible batch
processing, and integration into larger research workflows.

# Statement of need

Large-scale SAR processing often mixes mature external engines with custom glue code. In
practice, research groups need to chain ingestion, decoding or focusing, calibration,
geocoding, tiling, and post-processing while preserving metadata and keeping runs
reproducible across machines. SNAP provides a rich graph processing framework and
mission toolboxes [@snap], but research teams still need additional orchestration,
configuration management, chunked output handling, and automation-friendly interfaces.
`sarpyx` addresses this gap by wrapping end-to-end workflows in Python and CLI entry
points, while exposing the same building blocks to both interactive and batch use.

The target audience is remote-sensing researchers, SAR engineers, and platform teams
that operate preprocessing pipelines. The toolkit is especially useful when experiments
need both production-oriented orchestration and research-oriented derived products, such
as sub-aperture decomposition and tiling for downstream geospatial analytics or machine
learning. In the current public implementation, `sarpyx` exposes decode and focus
commands, a multi-step `worldsar` workflow, and Python interfaces for chunked product
access and export. By keeping these pieces in one package, the toolkit reduces
duplication between exploratory scripts and repeatable operational pipelines.

# State of the field

Several open tools address related needs, but with different emphases. SNAP [@snap] is
the underlying general-purpose processing platform and supplies many core operators.
`pyroSAR` focuses on large-scale SAR satellite data processing and analysis-ready
backscatter preparation [@truckenbrodt2019pyrosar; @truckenbrodt2019ard]. Open SAR
Toolkit emphasizes high-level Sentinel-1 inventory, download, and land-focused
preprocessing into analysis-ready products [@ost]. `snapista` is intentionally thin: it
provides a Pythonic layer around SNAP GPT graph creation and execution [@snapista]. The
ISCE/ISCE3 line targets interferometric and NISAR-oriented science processing with a
flexible scientific computing framework [@rosen2018isce3].

`sarpyx` was created because the research requirement here is not met by any single one
of these choices. Contributing only to a thin SNAP wrapper would not provide the
mission-aware command-line workflows, chunked array outputs, or integrated scientific
modules needed for this project. Building only on a Sentinel-1 ARD package would not
address the same combination of decode or focus stages, SNAP orchestration,
sub-aperture analysis, and tile-oriented exports. Reimplementing the mature
preprocessing operators already available in SNAP would also have been a poor use of
effort. The contribution of `sarpyx` is therefore the integration layer and research
abstraction: a Python toolkit that couples mature external operators to reproducible
orchestration, modular scientific extensions, and analysis-friendly outputs in one
workflow surface.

# Software design

Three design decisions shape `sarpyx`. First, the package adopts a hybrid architecture:
domain-specific orchestration is written in Python, while mature preprocessing steps are
delegated to SNAP where that is the most robust and maintainable choice. This avoids
re-implementing well-established operators, at the cost of carrying Java and SNAP
runtime requirements for some workflows. Second, the codebase is split into separate
layers for command-line interfaces, workflow orchestration, processing algorithms,
scientific metrics, and utility modules. This separation makes the project usable both
as an operator-facing CLI and as a library that can be imported into notebooks or
services.

Third, the toolkit emphasizes chunked outputs and tile-based processing. SAR scenes can
be large enough that monolithic in-memory handling becomes brittle for iterative
research, derived products, or downstream machine-learning pipelines. `sarpyx`
therefore exposes interfaces centered on chunked storage and selective slicing, and
includes parallel-friendly paths for large-product tiling. The trade-off is additional
I/O and metadata management complexity, but the gain is that the same outputs are easier
to reuse across experiments, quality-control routines, and downstream geospatial
systems. Together, these design choices favor reproducibility, extensibility, and
practical deployment over a minimal wrapper around external commands.

# Research impact statement

At minimum, `sarpyx` already demonstrates community-readiness signals through public
packaging, public documentation, and a reusable open repository with multi-author
development history. These properties matter for research software because they lower
the cost of installation, inspection, citation, and reuse.

The submitted repository includes a public documentation site, containerized execution
path, test suite, and reviewer smoke-test guide that make the current workflows
inspectable and repeatable from a fresh checkout. This is important for SAR research
software because the value of preprocessing and sub-aperture tooling depends on more
than algorithm availability: reviewers and downstream users must be able to reproduce
the software environment, locate the workflow entry points, and validate basic behavior
before applying the package to mission data. The JOSS review branch therefore includes
top-level citation, contribution, security, paper, bibliography, and reviewer-smoke-test
metadata alongside the software snapshot, so the reviewed artifact is directly reusable
and citable.

# AI usage disclosure

`sarpyx` development included limited use of generative AI assistance via GitHub
Copilot for selected code and repository-maintenance tasks. This submission pack was
also drafted with assistance from a generative AI system during paper and metadata
authoring. In all cases, the human authors defined the requirements, reviewed and
edited the generated material, executed tests or smoke checks where applicable, and
remained responsible for the technical correctness, scientific validity, licensing, and
final wording of the submission.

# Acknowledgements

The authors acknowledge support from ESA Φ-lab and the open-source SAR and geospatial
Python ecosystems that `sarpyx` builds upon.

# References
