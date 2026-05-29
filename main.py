import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
from liegroups.numpy import SE3

import camera as camera_module
import frameData
import invdepth_estimator_costVolume
import pose_estimator_gauss_newton


def pose_reader(fileName):
  matrix = np.loadtxt(fileName)
  if matrix.shape != (4, 4):
    raise ValueError(f"Expected a 4x4 pose matrix in {fileName}, got {matrix.shape}")
  return SE3.from_matrix(matrix, normalize=True)


def scene_index(path):
  return int(path.stem.split("_")[1].split(".")[0])


def dataset_scene_indices(dataset_dir):
  dataset_dir = Path(dataset_dir)
  image_indices = {
      scene_index(path) for path in (dataset_dir / "images").glob("scene_*.png")
  }
  pose_indices = {
      scene_index(path) for path in (dataset_dir / "poses").glob("scene_*.txt.new")
  }
  return sorted(image_indices & pose_indices)


def show_frame_state(frame, keyframe):
  cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
  cv2.imshow("frame", frame.image[1])
  cv2.namedWindow("invdepth", cv2.WINDOW_NORMAL)
  cv2.imshow("invdepth", keyframe.invDepth[1])
  cv2.namedWindow("invdepthVar", cv2.WINDOW_NORMAL)
  cv2.imshow("invdepthVar", np.sqrt(keyframe.invDepthVar[1]) * 10.0)
  cv2.waitKey(30)


def run(dataset_dir="dataset/desktop_dataset", max_frames=None, visualize=False):
  dataset_dir = Path(dataset_dir)
  width = 640
  height = 480
  fx = 481.20
  fy = 480.0
  cx = 319.5
  cy = 239.5

  frame = frameData.frameData()
  cam = camera_module.camera(fx, fy, cx, cy, width, height)
  pose_gn = pose_estimator_gauss_newton.pose_estimator_gauss_newton(
      cam, visualize=visualize
  )
  invdepth_estimator = invdepth_estimator_costVolume.invdepth_estimator_costVolume(
      cam, 1
  )
  keyframepose = SE3.identity()
  keyframe = None

  indices = dataset_scene_indices(dataset_dir)
  if max_frames is not None:
    indices = indices[:max_frames]

  for frame_number, imIndex in enumerate(indices):
    pose = pose_reader(dataset_dir / "poses" / f"scene_{imIndex:03d}.txt.new")
    image = cv2.imread(
        str(dataset_dir / "images" / f"scene_{imIndex:03d}.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
      raise FileNotFoundError(f"Could not read image for scene {imIndex:03d}")

    if frame_number == 0:
      keyframepose = copy.copy(pose)
      frame.setImage(image)
      keyframe = copy.deepcopy(frame)
      invDepth, invDepthVar = invdepth_estimator.getInvDepthAndVar()
      keyframe.setInvDepth(invDepth, invDepthVar)
    elif frame_number <= 3:
      frame.setImageAndPose(image, pose.dot(keyframepose.inv()))
      print("estimating invdepth")
      invdepth_estimator.update(frame, keyframe)
      print("done estimating invdepth")
      invDepth, invDepthVar = invdepth_estimator.getInvDepthAndVar()
      keyframe.setInvDepth(invDepth, invDepthVar)
    else:
      frame.setImage(image)
      print("estimating pose")
      pose_gn.optPose(frame, keyframe)
      print("done estimating pose")
      print("estimating invdepth")
      invdepth_estimator.update(frame, keyframe)
      print("done estimating invdepth")
      invDepth, invDepthVar = invdepth_estimator.getInvDepthAndVar()
      keyframe.setInvDepth(invDepth, invDepthVar)

    if visualize:
      show_frame_state(frame, keyframe)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="dataset/desktop_dataset")
  parser.add_argument("--max-frames", type=int)
  parser.add_argument("--visualize", action="store_true")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  run(dataset_dir=args.dataset, max_frames=args.max_frames, visualize=args.visualize)
