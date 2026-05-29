import numpy as np

import main


def test_pose_reader_loads_complete_homogeneous_matrix():
    pose = main.pose_reader("dataset/desktop_dataset/poses/scene_000.txt.new")
    matrix = pose.as_matrix()

    np.testing.assert_allclose(
        matrix,
        np.loadtxt("dataset/desktop_dataset/poses/scene_000.txt.new"),
        atol=1e-6,
    )
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_dataset_scene_indices_come_from_available_pairs():
    indices = main.dataset_scene_indices("dataset/desktop_dataset")

    assert indices == list(range(101))
