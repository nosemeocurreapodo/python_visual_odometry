import copy

import numpy as np
from liegroups.numpy import SE3

import camera as camera_module
import frameData
import params
import pose_estimator_gauss_newton


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


def make_frame_with_depth():
    y, x = np.indices((params.IMAGE_HEIGHT, params.IMAGE_WIDTH))
    image = (x + 2 * y).astype(np.float32)

    frame = frameData.frameData()
    frame.setImage(image)

    inv_depth = np.full((params.IMAGE_HEIGHT, params.IMAGE_WIDTH), 0.5, dtype=np.float32)
    inv_depth_var = np.full((params.IMAGE_HEIGHT, params.IMAGE_WIDTH), 0.01, dtype=np.float32)
    frame.setInvDepth(inv_depth, inv_depth_var)
    frame.pose = SE3.identity()
    return frame


def test_identity_pose_and_identical_images_have_zero_photometric_error(monkeypatch):
    use_small_images(monkeypatch)
    cam = make_camera()
    estimator = pose_estimator_gauss_newton.pose_estimator_gauss_newton(cam)
    keyframe = make_frame_with_depth()
    frame = copy.deepcopy(keyframe)

    residual, error_image = estimator.computeError(frame, keyframe, lvl=2)

    assert residual < 1e-12
    np.testing.assert_allclose(error_image, 0.0, atol=1e-12)


def test_pose_normal_equations_are_finite_symmetric_and_zero_residual_gives_zero_gradient(monkeypatch):
    use_small_images(monkeypatch)
    cam = make_camera()
    estimator = pose_estimator_gauss_newton.pose_estimator_gauss_newton(cam)
    keyframe = make_frame_with_depth()
    frame = copy.deepcopy(keyframe)

    gradient, hessian = estimator.computeHJPose(frame, keyframe, lvl=2)

    np.testing.assert_allclose(gradient, 0.0, atol=1e-7)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-7)
    assert np.all(np.isfinite(hessian))
    assert np.all(np.linalg.eigvalsh(hessian) >= -1e-6)
