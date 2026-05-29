import copy
from pathlib import Path

import cv2
import numpy as np
from liegroups.numpy import SE3

import camera as camera_module
import frameData
import invdepth_estimator_costVolume
import params
import pose_estimator_gauss_newton


DATASET = Path("dataset/desktop_dataset")
DATASET_POSE_TRANSLATION_ERROR_BASELINE = 0.010
DATASET_POSE_ROTATION_ERROR_BASELINE = 0.031


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


def use_small_dataset_images(monkeypatch):
    monkeypatch.setattr(params, "IMAGE_WIDTH", 128)
    monkeypatch.setattr(params, "IMAGE_HEIGHT", 96)


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


def load_dataset_pose(scene_index):
    pose_path = DATASET / "poses" / f"scene_{scene_index:03d}.txt.new"
    return SE3.from_matrix(np.loadtxt(pose_path), normalize=True)


def load_dataset_frame(scene_index):
    image_path = DATASET / "images" / f"scene_{scene_index:03d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None

    frame = frameData.frameData()
    frame.setImage(image)
    return frame


def pose_error(actual, expected):
    delta = actual.dot(expected.inv()).as_matrix()
    translation_error = np.linalg.norm(delta[:3, 3])
    rotation_error = np.arccos(
        np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    )
    return translation_error, rotation_error


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


def test_dataset_pose_estimate_error_does_not_regress(monkeypatch):
    use_small_dataset_images(monkeypatch)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: None)

    cam = make_camera()
    depth_estimator = invdepth_estimator_costVolume.invdepth_estimator_costVolume(
        cam, lvl=1
    )
    pose_estimator = pose_estimator_gauss_newton.pose_estimator_gauss_newton(cam)

    keyframe_pose = load_dataset_pose(0)
    keyframe = load_dataset_frame(0)
    inv_depth, inv_depth_var = depth_estimator.getInvDepthAndVar()
    keyframe.setInvDepth(inv_depth, inv_depth_var)

    for scene_index in range(1, 4):
        frame = load_dataset_frame(scene_index)
        ground_truth_pose = load_dataset_pose(scene_index).dot(keyframe_pose.inv())
        frame.pose = copy.copy(ground_truth_pose)
        depth_estimator.update(frame, keyframe)
        inv_depth, inv_depth_var = depth_estimator.getInvDepthAndVar()
        keyframe.setInvDepth(inv_depth, inv_depth_var)

    target_scene_index = 4
    target_frame = load_dataset_frame(target_scene_index)
    expected_pose = load_dataset_pose(target_scene_index).dot(keyframe_pose.inv())

    pose_estimator.optPose(target_frame, keyframe)

    translation_error, rotation_error = pose_error(target_frame.pose, expected_pose)
    assert translation_error <= DATASET_POSE_TRANSLATION_ERROR_BASELINE
    assert rotation_error <= DATASET_POSE_ROTATION_ERROR_BASELINE
