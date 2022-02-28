import cv2 
import numpy as np
import params
from liegroups.numpy import SE3

import common

class invdepth_estimator_costVolume:
  def __init__(self, camera, lvl):
    self.camera = camera
    self.lvl = lvl
    self.numInvDepths = 32
    self.minInvDepth = 0.001
    self.maxInvDepth = 1.5
    self.invDepthStep = (self.maxInvDepth - self.minInvDepth)/(self.numInvDepths-1)
    
    self.invDepths = np.zeros((self.numInvDepths))
    for z in range(0, self.numInvDepths):
      self.invDepths[z] = self.invDepthStep*z + self.minInvDepth
    
    self.costVolume = np.zeros((self.camera.height[self.lvl],self.camera.width[self.lvl],self.numInvDepths), dtype = np.float32)
    self.obsCount = np.zeros((self.camera.height[self.lvl],self.camera.width[self.lvl],self.numInvDepths), dtype = np.int32)
    
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):
        maxz = self.numInvDepths*0.75
        minz = self.numInvDepths*0.25
        z = int((maxz-minz) * float(y)/(self.camera.height[self.lvl]-1.0) + minz)
        self.obsCount[y,x,z] = 1

  def getInvDepthAndVar(self):
    invDepth = np.zeros((self.camera.height[self.lvl],self.camera.width[self.lvl]), dtype = np.float32)
    invDepthVar = (10**2)*np.ones((self.camera.height[self.lvl],self.camera.width[self.lvl]), dtype = np.float32)
        
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):
        bestInvDepthCost = 100000000.0
        bestInvDepth = -1
        worstInvDepthCost = 0.0
        worstInvDepth = -1
        for z in range(0, self.numInvDepths):
          cost = self.costVolume[y,x,z]
          count = self.obsCount[y,x,z]
          meancost = 0
          if count > 0:
            meancost = cost/count 
          else:
            continue
          if meancost < bestInvDepthCost:
            bestInvDepthCost = meancost
            bestInvDepth = self.invDepths[z]
          if meancost > worstInvDepthCost:
            worstInvDepthCost = meancost
            worstInvDepth = self.invDepths[z]
        if bestInvDepth == worstInvDepth:
          continue
        #print("pixel ", x, " ", y)
        #print("invDepth ", bestInvDepth)
        invDepth[y,x] = bestInvDepth
        invDepthVar[y,x] = 1.0#self.invDepthStep**2# (abs(bestInvDepth - worstInvDepth)/3.0)**2
        
    filteredInvDepth = invDepth# common.filterInvDepthWithVar(invDepth, invDepthVar, self.camera, self.lvl)
    return [filteredInvDepth, invDepthVar]
          
  def setInvDepth(self, invDepth):
    invDepthResized = cv2.resize(invDepth, (self.camera.width[self.lvl], self.camera.height[self.lvl]))
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):
        for z in range(0, self.numInvDepths):
          idepth = self.invDepths[z]
          self.costVolume[y,x,z] = (invDepthResized[y,x] - idepth)**2
          self.obsCount[y,x,z] = 1

  def filterInvDepth(self):
    for y in range(1, self.camera.height[self.lvl]-1):
      for x in range(1, self.camera.width[self.lvl]-1):
        
        m = self.mu[y,x]
        if m < 0.0:
          continue
        
        new_m = 0.0
        total_s = 0.0
        for yi in range(-1,2):
          for xi in range(-1,2):
            current_m = self.mu[y+yi,x+xi]
            current_s = self.sigma[y+yi,x+xi]
            if current_m < 0.0:
              continue
            if abs(m - current_m) > 1.0*np.sqrt(current_s):
              continue
            weight = 1.0/current_s
            new_m = new_m + current_m*weight
            total_s = total_s + weight
 
        self.mu[y,x] = new_m/total_s


    
  def update(self, frame, keyframe):
  
    lvl = self.lvl
    width = self.camera.width[lvl]
    height = self.camera.height[lvl]
    fx = self.camera.fx[lvl]
    fy = self.camera.fy[lvl]
    cx = self.camera.cx[lvl]
    cy = self.camera.cy[lvl]
    fxinv = self.camera.fxinv[lvl]
    fyinv = self.camera.fyinv[lvl]
    cxinv = self.camera.cxinv[lvl]
    cyinv = self.camera.cyinv[lvl]

    relativePose = frame.pose.dot(keyframe.pose.inv())
     
    for y in range(1, height-1):
      for x in range(1, width-1):

        kfpixel = np.array([x,y])
        kfray = np.array([fxinv*kfpixel[0] + cxinv, fyinv*kfpixel[1] + cyinv, 1.0])
          
        for z in range(0, self.numInvDepths):
        
          invDepth = self.invDepths[z]  

          kfpoint = kfray/invDepth
          
          fpoint = relativePose.dot(kfpoint)
          
          if fpoint[2] <= 0.0:
            continue
          
          fray = fpoint/fpoint[2]
          fpixel = np.array([fx*fray[0] + cx, fy*fray[1] + cy])

          if fpixel[0] < 1.0 or fpixel[0] >= width-1 or fpixel[1] < 1.0 or fpixel[1] >= height-1:
            continue

          residual = 0.0
          for yi in range(-1,2):
            for xi in range(-1,2):
              kf = keyframe.image[self.lvl][y+yi,x+xi]
              f = frame.image[self.lvl][int(fpixel[1])+yi,int(fpixel[0])+xi]
              #f = common.getSubPixelValue(frame.image[lvl], fpixel+np.array([xi,yi]))
              residual += (float(f)-float(kf))**2
          """    
          kf = keyframe.image[lvl][y,x]
          f = common.getSubPixelValue(frame.image[lvl], fpixel)
          residual = (float(f)-float(kf))**2
          """
          
          #print("voxel: ", y, " ", x, " ", z)
          #print("residual: ", residual)
          self.costVolume[y,x,z] += residual
          self.obsCount[y,x,z] += 1
