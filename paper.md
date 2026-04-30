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

Synthetic aperture radar (SAR) satellites provide all-weather, day-and-night observations
of the Earth. These data are used in applications such as land monitoring, disaster
response, ocean surveillance, cryosphere analysis, and infrastructure assessment.
However, transforming SAR products into analysis-ready data often requires a sequence of
mission-specific tools, processing graphs, shell scripts, and custom notebook code. This
fragmentation makes experiments difficult to reproduce and operational workflows
difficult to maintain.

`sarpyx` is a Python package for building reproducible SAR processing workflows. It
combines command-line entry points with importable Python modules so that the same
processing components can be used in local experiments, batch workflows, and larger
research systems. The package supports decoding and focusing workflows, SNAP-backed
preprocessing, tiled export of large products, chunked array access, and sub-aperture
decomposition. Sub-aperture decomposition is useful for studying directional scattering
behaviour, azimuth-dependent signatures, and motion-sensitive SAR observables.

The package is designed for researchers and engineers who need to move from raw or
higher-level SAR products to reusable outputs without rewriting orchestration code for
each experiment. Its implementation emphasizes environment-based configuration,
container-compatible execution, modular processing interfaces, and data products that
can be inspected, sliced, exported, and reused across downstream geospatial or machine
learning workflows.

# Statement of need

SAR processing workflows commonly combine mature external engines with project-specific
automation. A typical workflow may involve product ingestion, decoding or focusing,
radiometric processing, geocoding, metadata preservation, tiling, and export to formats
suitable for later analysis. In many research environments these steps are implemented
through a mixture of SNAP graphs, scripts, notebooks, and local conventions. This makes
it hard to reproduce results across machines, compare experiments consistently, or
transfer prototype code into operational processing.

SNAP provides a broad and mature processing framework for Earth observation data
[@snap], but many research teams still need additional tooling around workflow
orchestration, configuration management, chunked output handling, product tiling, and
automation-friendly execution. `sarpyx` addresses this integration gap. It does not
attempt to replace mature SAR operators already available in SNAP. Instead, it exposes a
Python and CLI workflow layer that coordinates external processing, scientific
extensions, and analysis-oriented outputs.

The target users are remote-sensing researchers, SAR algorithm developers, and platform
teams that operate reproducible preprocessing pipelines. The package is particularly
useful when the same project requires both production-oriented orchestration and
research-oriented derived products, such as sub-aperture stacks or tiled exports for
downstream geospatial analytics. In the current implementation, `sarpyx` provides
decode and focus commands, a multi-step `worldsar` workflow, and Python interfaces for
chunked product access and export. Keeping these capabilities in one package reduces
duplication between exploratory scripts, reviewer-facing examples, and repeatable
processing pipelines.

# State of the field

Several open-source projects address related parts of the SAR processing ecosystem.
SNAP is the underlying general-purpose processing platform and provides many core
operators used by the community [@snap]. `pyroSAR` supports large-scale SAR data
processing and analysis-ready backscatter preparation, with particular emphasis on
automated handling of SAR satellite data and integration with geospatial processing
workflows [@truckenbrodt2019pyrosar; @truckenbrodt2019ard]. The Open SAR Toolkit
focuses on Sentinel-1 inventory, download, and land-oriented preprocessing into
analysis-ready data products [@ost]. `snapista` provides a Pythonic interface for
constructing and executing SNAP GPT graphs [@snapista]. ISCE and ISCE3 support
interferometric SAR and NISAR-oriented science processing through a flexible scientific
computing framework [@rosen2018isce3].

`sarpyx` occupies a different position in this landscape. It is not only a SNAP graph
wrapper, and it is not limited to Sentinel-1 analysis-ready backscatter generation. Its
purpose is to provide mission-aware command-line workflows, Python-accessible
processing modules, chunked product interfaces, tiled exports, and research modules such
as sub-aperture analysis within one reproducible workflow surface.

The main build-versus-contribute justification is therefore architectural. Contributing
only to a thin SNAP wrapper would not provide the higher-level orchestration,
mission-aware workflow entry points, chunked storage abstractions, and scientific
modules required by this project. Building solely on an existing Sentinel-1 ARD package
would not cover the same combination of decode or focus stages, SNAP orchestration,
sub-aperture decomposition, and tile-oriented export. Conversely, reimplementing mature
SNAP operators would duplicate well-tested functionality and increase maintenance
burden. `sarpyx` instead contributes an integration and research abstraction layer that
connects mature external processing with reproducible Python workflows and
analysis-ready downstream products.

# Software design

The design of `sarpyx` is based on three architectural choices.

First, the package uses a hybrid processing model. Workflow orchestration,
configuration, data access, and scientific extensions are implemented in Python, while
selected preprocessing stages are delegated to SNAP when mature SNAP operators are the
most robust choice. This design avoids unnecessary reimplementation of established SAR
processing steps. The cost is an additional runtime dependency on Java and SNAP for
workflows that require those operators. The benefit is that `sarpyx` can combine
well-established external processing with reproducible Python automation and
experiment-specific extensions.

Second, the package separates command-line interfaces, workflow orchestration,
processing algorithms, scientific metrics, and utility layers. This separation allows
the same functionality to be used in different execution contexts. Operators can run
predefined workflows through the CLI, while researchers can import the same modules in
notebooks, services, or batch-processing code. This structure also reduces coupling
between mission-specific workflow logic and reusable processing components.

Third, `sarpyx` emphasizes chunked outputs and tile-based processing. SAR scenes are
often too large for brittle monolithic in-memory workflows, especially when generating
derived products, running repeated quality checks, or preparing data for machine
learning pipelines. The package therefore exposes interfaces for chunked storage,
selective slicing, and parallel-friendly export. This introduces additional metadata and
I/O-management complexity, but it makes outputs easier to inspect, reuse, and integrate
with downstream geospatial systems.

These choices prioritize reproducibility, extensibility, and operational reuse over a
minimal command wrapper. They also support the research use case that motivated the
package: moving consistently from mission data and external processing engines to
derived SAR products that can be analysed, validated, and reused across experiments.

# Research impact statement

The current impact of `sarpyx` is supported by both a reusable public implementation and
active use in collaborative SAR research. The software has been used in the ESA
WORLDSAR project, carried out in collaboration with NASA and DLR, where reproducible SAR
processing, workflow orchestration, and derived-product generation are required across
institutional and technical boundaries.

The repository provides installable software, public documentation, container-compatible
execution, tests, and reviewer smoke-test material. These elements make the package
inspectable and repeatable from a fresh checkout, which is essential for SAR processing
software where scientific value depends on both algorithmic behavior and reproducible
execution.

The submitted JOSS branch includes citation, contribution, security, paper,
bibliography, and reviewer-smoke-test metadata alongside the software snapshot. This
helps reviewers and downstream users identify the workflow entry points, reproduce the
software environment, and verify basic behavior before applying the package to mission
data. The package also provides a practical bridge between interactive research code and
repeatable processing workflows, reducing the cost of moving SAR experiments from local
prototypes to reusable pipelines.

By combining SNAP-backed processing, Python workflow orchestration, chunked data access,
tiled export, and sub-aperture analysis, `sarpyx` supports research groups that need
more than isolated operators but less than a fully bespoke processing platform. Its
research value is in making SAR preprocessing and derived-product generation more
reproducible, inspectable, and reusable across experiments and collaborative projects.

# AI usage disclosure

Development of `sarpyx` included limited use of generative AI assistance through GitHub
Copilot for selected coding and repository-maintenance tasks. Generative AI assistance
was also used during preparation of the JOSS submission materials, including drafting and
editing of paper and metadata text.

In all cases, the human authors defined the requirements, reviewed and edited generated
material, ran tests or smoke checks where applicable, and retained responsibility for
technical correctness, scientific validity, licensing, and final wording.

# Acknowledgements

The authors acknowledge support from ESA Φ-lab and from the open-source SAR, Earth
observation, and geospatial Python communities that `sarpyx` builds upon.

# References