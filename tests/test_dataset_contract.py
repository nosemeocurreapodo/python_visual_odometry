from pathlib import Path

import numpy as np


DATASET = Path("dataset/desktop_dataset")


def _scene_index(path):
    return int(path.name.split(".")[0].split("_")[1])


def test_dataset_images_and_poses_are_paired_and_contiguous():
    image_ids = {_scene_index(path) for path in (DATASET / "images").glob("scene_*.png")}
    pose_ids = {_scene_index(path) for path in (DATASET / "poses").glob("scene_*.txt.new")}

    assert image_ids == pose_ids
    assert image_ids == set(range(min(image_ids), max(image_ids) + 1))


def test_pose_files_are_valid_homogeneous_se3_matrices():
    for pose_path in sorted((DATASET / "poses").glob("scene_*.txt.new")):
        pose = np.loadtxt(pose_path)
        rotation = pose[:3, :3]

        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-4)
