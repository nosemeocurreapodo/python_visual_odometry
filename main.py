import cv2 
import sys
import numpy as np
import copy
from liegroups.numpy import SE3

import camera
import frameData
import pose_estimator_gauss_newton
import invdepth_estimator_kalman
import invdepth_estimator_costVolume

def pose_reader(fileName):
  poseFile = open(fileName, "r")
  flines = poseFile.readlines()

  m = np.zeros((4,4))
  
  for row in range(0,len(flines)-1):
  
    words = flines[row].split()
  
    m[row,0] = float(words[0])
    m[row,1] = float(words[1])
    m[row,2] = float(words[2])
    m[row,3] = float(words[3])  
    
  #imageFileName = "../../../Trajectory_30_seconds/scene_%03d.png" % (imIndex)
  #print(imageFileName)

  #print("mat")
  #print(m)
  pose = SE3.from_matrix(m,normalize=True)
  #print(pose)
  return pose
  
width = 640
height = 480
fx = 481.20
fy = 480.0
cx = 319.5
cy = 239.5

frame = frameData.frameData()
camera = camera.camera(fx,fy,cx,cy,width,height)
pose_gn = pose_estimator_gauss_newton.pose_estimator_gauss_newton(camera)
#invdepth_estimator = invdepth_estimator_kalman.invdepth_estimator_kalman(camera, 1)
invdepth_estimator = invdepth_estimator_costVolume.invdepth_estimator_costVolume(camera,1)
keyframepose = SE3.identity() 
      
for imIndex in range(0,500):

  pose = pose_reader("../../../desktop_dataset_2/scene_%03d.txt.new" % (imIndex))
  image = cv2.imread("../../../desktop_dataset_2/scene_%03d.png" % (imIndex), cv2.IMREAD_GRAYSCALE)
  
  if imIndex == 0:
    keyframepose = copy.copy(pose)
    #frame.setImageAndPose(image, pose)
    frame.setImage(image)
    keyframe = copy.deepcopy(frame)
    #fs = cv2.FileStorage("../../../desktop_dataset_2/scene_depth_000.yml", cv2.FILE_STORAGE_READ)
    #invdepth_estimator.setInvDepth(fs.getNode("idepth").mat())
    [invDepth, invDepthVar] = invdepth_estimator.getInvDepthAndVar()
    keyframe.setInvDepth(invDepth, invDepthVar)
  if imIndex > 0 and imIndex <= 3:
    frame.setImageAndPose(image, pose.dot(keyframepose.inv()))
    print("estimating invdepth")
    invdepth_estimator.update(frame, keyframe)
    print("done estimating invdepth")
    [invDepth, invDepthVar] = invdepth_estimator.getInvDepthAndVar()
    keyframe.setInvDepth(invDepth, invDepthVar)   
  if imIndex > 3:
    frame.setImage(image)
    print("estimating pose")
    pose_gn.optPose(frame, keyframe)
    print("done estimating pose")
    #print("real pose: ", pose)
    #print("est pose: ", frame.pose)
    print("estimating invdepth")
    invdepth_estimator.update(frame, keyframe)
    print("done estimating invdepth")
    [invDepth, invDepthVar] = invdepth_estimator.getInvDepthAndVar()
    keyframe.setInvDepth(invDepth, invDepthVar) 


  cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
  cv2.imshow("frame", frame.image[1])
  cv2.namedWindow("invdepth", cv2.WINDOW_NORMAL);
  invdepthNormalized = keyframe.invDepth[1]# (keyframe.invDepth[1] - np.amin(keyframe.invDepth[1]))/(np.amax(keyframe.invDepth[1]) - np.amin(keyframe.invDepth[1]))
  cv2.imshow("invdepth", invdepthNormalized)
  cv2.namedWindow("invdepthVar", cv2.WINDOW_NORMAL);
  invdepthVarNormalized = np.sqrt(keyframe.invDepthVar[1])*10.0#(keyframe.invDepthVar[1] - np.amin(keyframe.invDepthVar[1]))/(np.amax(keyframe.invDepthVar[1]) - np.amin(keyframe.invDepthVar[1]))
  cv2.imshow("invdepthVar", invdepthVarNormalized)
  cv2.waitKey(30)
