=========
Changelog
=========

All notable changes to **de-twin** are documented here.

Fragment files in ``upcoming_changes/`` are assembled into this file by
`towncrier <https://towncrier.readthedocs.io/>`_ when a release is prepared
(see ``upcoming_changes/README.rst`` for contributor instructions).

.. towncrier release notes start

0.1.1 (2026-09-24)
==================

Bug Fixes
---------

- The deapi face reports "Specimen Pixel Size X/Y (nanometers)" per binned pixel, including hardware and software binning, as DE-Server does.
- The deapi face treats ``start_acquisition(0)`` as live view (repeat until stopped), as
  DE-Server does, instead of a single acquisition; and repeated acquisitions no longer repeat
  the same detector noise, because noise is now seeded by a twin-wide frame counter rather than
  the frame index within each request.


0.1.0 (2026-09-24)
==================

New Features
------------

- First release of the de-twin digital twin: a simulated TEM column (JEOL/TFS-style
  modes, magnification and camera-length ladders, stage motion, aberrations and an
  optional probe/image corrector), seeded specimens (grids, thin films, proteins in ice,
  FIB lamellae, an in-situ heating chip) with real crystal orientations from diffsims and
  orix, wave-optical TEM imaging, diffraction and coherent 4D-STEM, and a detector model
  for every Direct Electron camera. ``DigitalTwin`` runs it all in-process.
- The twin can stand in for real software: a DE-TEM-Channel SOAP server
  (``de-twin serve --soap``) that DE-Server, de_microscope, de_autopilot and
  de_ground_crew drive unchanged, a deapi-compatible fake DE-Server
  (``--deapi``), and a shared-memory frame source that feeds a real DE-Server
  through its processing pipeline (``--shm``, test pattern "External Frame Source
  (Shared Memory)"). It can also follow a real or Dummy DE-TEM-Channel and a DENS
  Impulse holder instead of simulating them.


Bug Fixes
---------

- Coherent 4D-STEM data no longer depends on the machine: the probe's partial-coherence
  modes were truncated through a degenerate pair of eigenmodes, whose basis LAPACK chooses
  differently with BLAS threading, so the same request gave different patterns on different
  computers. Truncation now keeps degenerate groups whole.


Maintenance
-----------

- Packaging, documentation site and release workflows (towncrier changelog,
  Prepare Release PR, PyPI publishing when a GitHub Release is published).
- Raised the minimum supported versions to numpy 2.0, scipy 1.13 and diffpy.structure 3.2
  (and matplotlib 3.9 for the docs), the oldest set the test suite passes with. Timing
  budgets in the performance tests now scale with ``DE_TWIN_PERF_SLACK`` so shared CI
  runners can check them without being tuned to one workstation.
