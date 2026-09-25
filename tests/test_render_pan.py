"""Panning: a stage move inside the render margin is a crop, not a render — and the same image."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.render.config import RenderConfig
from de_twin.twin import DigitalTwin


def _pair(name="Dense Au on holey C"):
    pan = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=3)
    ref = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=3,
                      render_config=RenderConfig(pan_margin=0.0))
    return pan, ref


def _move(tws, dx_um, dy_um=0.0):
    for tw in tws:
        s = tw.column.state().stage
        tw.column.move_stage(x=s.x_um + dx_um, y=s.y_um + dy_um)


@pytest.mark.parametrize("dx_px,dy_px", [(7, 0), (0, -30), (90, 60)])
def test_a_panned_view_is_the_full_render(dx_px, dy_px):
    pan, ref = _pair()
    req = pan.request()
    pan.flux(req)
    px = pan.optics(req).specimen_pixel_nm / 1000.0
    _move((pan, ref), dx_px * px, dy_px * px)
    a, b = pan.flux(req), ref.flux(req)
    inner = (slice(40, -40), slice(40, -40))  # away from the transfer's periodic edges
    assert np.corrcoef(a[inner].ravel(), b[inner].ravel())[0, 1] > 0.99
    assert a.mean() == pytest.approx(b.mean(), rel=0.02)


def test_a_nudge_is_a_crop_and_a_jump_renders():
    pan, _ = _pair()
    req = pan.request()
    pan.flux(req)
    built = pan.renderer.rasters_built
    px = pan.optics(req).specimen_pixel_nm / 1000.0
    _move((pan,), 50 * px)
    pan.flux(req)
    assert pan.renderer.rasters_built == built, "inside the margin: no new raster"
    _move((pan,), 3000 * px)
    pan.flux(req)
    assert pan.renderer.rasters_built == built + 1, "beyond it: rendered again"


def test_the_frames_of_an_exposure_share_one_array():
    pan, _ = _pair()
    req = pan.request()
    a, b = pan.flux(req), pan.flux(req)
    assert a is b and not a.flags.writeable


def test_an_exposure_renders_once_on_a_running_clock():
    """The clock advances every frame; a static specimen must not re-finish its image."""
    from de_twin.clock import ManualClock as MC

    clock = MC()
    tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=clock, seed=3)
    req = tw.request(frame_time_s=0.025, total_frames=10)
    list(tw.frames(tw.request(frame_time_s=0.025, total_frames=1)))
    before = tw.renderer.frames_from_cache
    list(tw.frames(req))
    assert tw.renderer.frames_from_cache - before >= 9
