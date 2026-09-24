"""Test configuration.

``RenderConfig.stem_model`` defaults to "auto", which picks the coherent 4D-STEM model for
small (<= 256^2) patterns or ptychography-like scans. The STEM tests written for the
kinematic model (disk caches, intensity jitter, probe-footprint blend, descan ramp, virtual
image speed) exercise that model on such geometries, so outside the coherent test modules
"auto" resolves to "kinematic" here. Explicit ``stem_model="coherent"`` is never overridden.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _auto_stem_model_is_kinematic_for_legacy_tests(request, monkeypatch):
    name = request.module.__name__.rsplit(".", 1)[-1]
    if name.startswith("test_coherent"):
        yield
        return
    from de_twin.render import coherent

    original = coherent.choose_model

    def choose(optics, cfg, sampling_fn):
        if str(cfg.stem_model).lower() == "auto":
            return "kinematic"
        return original(optics, cfg, sampling_fn)

    monkeypatch.setattr(coherent, "choose_model", choose)
    import de_twin.render.renderer as renderer
    monkeypatch.setattr(renderer, "choose_model", choose)
    yield
