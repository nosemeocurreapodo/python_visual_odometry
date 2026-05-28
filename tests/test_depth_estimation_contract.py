import copy

import numpy as np
import pytest

import camera as camera_module
import frameData
import invdepth_estimator_costVolume
import params


def make_camera():
    return camera_module.camera(
        fx=481.20,
        fy=480.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )


def use_small_images(monkeypatch):
    monkeypatch.setattr(params, "IMAGE_WIDTH", 64)
    monkeypatch.setattr(params, "IMAGE_HEIGHT", 48)


def make_textured_frame():
    y, x = np.indices((params.IMAGE_HEIGHT, params.IMAGE_WIDTH))
    image = ((3 * x + 5 * y) % 251).astype(np.uint8)
    frame = frameData.frameData()
    frame.setImage(image)
    return frame


def test_cost_volume_depth_outputs_match_estimator_level_shape(monkeypatch):
    use_small_images(monkeypatch)
    cam = make_camera()
    estimator = invdepth_estimator_costVolume.invdepth_estimator_costVolume(cam, lvl=1)

    inv_depth, inv_depth_var = estimator.getInvDepthAndVar()

    assert inv_depth.shape == (cam.height[1], cam.width[1])
    assert inv_depth_var.shape == (cam.height[1], cam.width[1])
    assert np.all(np.isfinite(inv_depth))
    assert np.all(np.isfinite(inv_depth_var))
    assert np.all(inv_depth_var > 0.0)


@pytest.mark.xfail(reason="A monocular depth estimator should reject zero-baseline observations.")
def test_zero_baseline_update_does_not_accumulate_depth_evidence(monkeypatch):
    use_small_images(monkeypatch)
    cam = make_camera()
    estimator = invdepth_estimator_costVolume.invdepth_estimator_costVolume(cam, lvl=1)
    keyframe = make_textured_frame()
    frame = copy.deepcopy(keyframe)

    obs_count_before = estimator.obsCount.copy()
    estimator.update(frame, keyframe)

    np.testing.assert_array_equal(estimator.obsCount, obs_count_before)
