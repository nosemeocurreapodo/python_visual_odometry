import cv2 
import numpy as np
import params
from liegroups.numpy import SE3

import common

class invdepth_estimator_kalman:
  def __init__(self, camera, lvl):
    self.camera = camera
    self.lvl = lvl

    
    self.mu = np.ones((self.camera.height[self.lvl],self.camera.width[self.lvl]), dtype = np.float32)*0.5
    
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):
        self.mu[y,x] = 0.1 + (1.0-0.1) * float(y)/self.camera.height[self.lvl]
    
    #self.mu = np.random.rand(self.camera.height[self.lvl], self.camera.width[self.lvl]) + 0.5
    self.sigma = np.ones((self.camera.height[self.lvl],self.camera.width[self.lvl]), dtype = np.float32)*(0.25**2)

  def getInvDepthAndVar(self):
    return [self.mu,self.sigma]
    
  def setInvDepth(self, invDepth):
    self.mu = cv2.resize(invDepth, (self.camera.width[self.lvl], self.camera.height[self.lvl]))
    
  def removeOutliers(self, invDepth, invDepthVar):
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):

        idepth = invDepth[y,x]
        var = invDepthVar[y,x]
        
        if idepth <= 0.0:
          continue
        
        support = 0
        for yi in range(-1,2):
          for xi in range(-1,2):
            nidepth = invDepth[y+yi,x+xi]
            nvar = invDepthVar[y+yi,x+xi]
            if nidepth < 0.0:
              continue
            if abs(idepth - nidepth) < 1.0*np.sqrt(var) and abs(idepth - nidepth) < 1.0*np.sqrt(nvar):
              support += 1
        if support < 3:
          invDepth[y,x] = -1
      
  def epiLineSearh(self, frame, keyframe):

    width = self.camera.width[self.lvl]
    height = self.camera.height[self.lvl]
    fx = self.camera.fx[self.lvl]
    fy = self.camera.fy[self.lvl]
    cx = self.camera.cx[self.lvl]
    cy = self.camera.cy[self.lvl]
    fxinv = self.camera.fxinv[self.lvl]
    fyinv = self.camera.fyinv[self.lvl]
    cxinv = self.camera.cxinv[self.lvl]
    cyinv = self.camera.cyinv[self.lvl]

    epilineInvDepth = -1.0*np.ones((height, width), dtype = np.float32)
    epilineVar = -1.0*np.ones((height, width), dtype = np.float32)
    
    relativePose = frame.pose.dot(keyframe.pose.inv())
    
    for y in range(1, height-1):
      for x in range(1, width-1):
      
        #if np.sqrt(keyframe.imageDerivative[self.lvl][y,x][0]*keyframe.imageDerivative[self.lvl][y,x][0] + keyframe.imageDerivative[self.lvl][y,x][1]*keyframe.imageDerivative[self.lvl][y,x][1]) < 64.0:
        #  continue
      
        kfpixel = np.array([x,y]) 
        kfray = np.array([fxinv*kfpixel[0] + cxinv, fyinv*kfpixel[1] + cyinv, 1.0]) 
                
        minInvDepth = self.mu[y,x] - 3.0*np.sqrt(self.sigma[y,x])
        if minInvDepth < 0.0001:
          minInvDepth = 0.0001
        maxInvDepth = self.mu[y,x] + 3.0*np.sqrt(self.sigma[y,x])

        #compute epipolar line width, so we can compute the step in inverse depth
        #first, compute the position of the pixel in the frame corresponding to the point with maximum invdepth
        kfMaxPoint = kfray/maxInvDepth
        fMaxPoint = relativePose.dot(kfMaxPoint)
        fMaxRay = fMaxPoint/fMaxPoint[2]
        fMaxPixel = np.array([fx*fMaxRay[0] + cx, fy*fMaxRay[1] + cy])
        if fMaxPixel[0] < 0.0:
          fMaxPixel[0] = 0.0
        if fMaxPixel[0] > width:
          fMaxPixel[0] = width
        if fMaxPixel[1] < 0.0:
          fMaxPixel[1] = 0.0
        if fMaxPixel[1] > height:
          fMaxPixel[1] = height
          
        #now compute the pixel postion of the point corresponding with the minimum invdepth
        kfMinPoint = kfray/minInvDepth
        fMinPoint = relativePose.dot(kfMinPoint)
        fMinRay = fMinPoint/fMinPoint[2]
        fMinPixel = np.array([fx*fMinRay[0] + cx, fy*fMinRay[1] + cy])
        if fMinPixel[0] < 0.0:
          fMinPixel[0] = 0.0
        if fMinPixel[0] > width:
          fMinPixel[0] = width
        if fMinPixel[1] < 0.0:
          fMinPixel[1] = 0.0
        if fMinPixel[1] > height:
          fMinPixel[1] = height
          
        #compute distance in pixels, wich correspond to the maximum steps!                  
        maxSteps = 1.0*np.sqrt((fMaxPixel[0]-fMinPixel[0])**2+(fMaxPixel[1]-fMinPixel[1])**2)
        
        if maxSteps < 3:
          continue
        
        invDepthStep = (maxInvDepth - minInvDepth)/maxSteps

	#lets make the standar deviation of the measurement as 1 times the inverse depth step (corresponding to 1 pixels!)
        epilineVar[y,x] = (1.0*1.0*invDepthStep)**2
        #epilineVar[y,x] = (0.25*(maxInvDepth - minInvDepth))**2

        kfInvDepth = minInvDepth               
        bestResidual = 100000.0
        steps = 0
        while(steps < maxSteps):
        
          kfInvDepth += invDepthStep
          steps += 1
          
          if kfInvDepth <= 0.0:
            #epilinedepth[y,x] = -1.0
            #break
            continue
                      
          if kfInvDepth > maxInvDepth:
            break
     
          kfpoint = kfray/kfInvDepth
          fpoint = relativePose.dot(kfpoint)

          #if point behind camera, continue
          if fpoint[2] <= 0.0:
            #epilinedepth[y,x] = -1.0
            #break
            continue

          fray = fpoint/fpoint[2]
          fpixel = np.array([fx*fray[0] + cx, fy*fray[1] + cy])
        
          #if pixel outside image, continue
          if fpixel[0] < 2.0 or fpixel[0] >= width-2 or fpixel[1] < 2.0 or fpixel[1] >= height-2:
            #epilineInvDepth[y,x] = -1.0
            #break
            continue
        
          residual = 0.0
          
          
          for yi in range(-1,2):
            for xi in range(-1,2):
              kf = keyframe.image[self.lvl][y+yi,x+xi]
              f = frame.image[self.lvl][int(fpixel[1])+yi,int(fpixel[0])+xi]
              #f = common.getSubPixelValue(frame.image[self.lvl], fpixel+np.array([xi,yi]))
              residual += (float(f)-float(kf))**2
          
          """
          kf = keyframe.image[self.lvl][y,x]
          f = frame.image[self.lvl][int(round(fpixel[1])),int(round(fpixel[0]))]
          residual += (float(f)-float(kf))**2
          """
          """
          kf = keyframe.image[self.lvl][y,x]
          f = common.getSubPixelValue(frame.image[self.lvl], fpixel)
          residual += (float(f)-float(kf))**2
          """           
          if residual < bestResidual:
            bestResidual = residual
            epilineInvDepth[y,x] = kfInvDepth
        
        if x == int(width*0.5) and y == int(height*0.5):
          print("pixel ",x," ",y)
          print("steps ", steps)  
          print("invDepthStep ", invDepthStep)
          print("mu ", self.mu[y,x])
          print("sigma ", self.sigma[y,x])
          print("max-min invDepth: ", maxInvDepth, " ", minInvDepth)
          print("epiInvDepth ", epilineInvDepth[y,x])   
          print("epiVar ", epilineVar[y,x])        
          #print("best depth for ", x, " ", y, " ", epilinedepth[y,x])
        
    return [epilineInvDepth, epilineVar]
    
  def update(self, frame, keyframe):
    
    [sensorInvDepth, sensorVar] = self.epiLineSearh(frame, keyframe)
    self.removeOutliers(sensorInvDepth, sensorVar)
    
    for y in range(0, self.camera.height[self.lvl]):
      for x in range(0, self.camera.width[self.lvl]):

        #prediction
        mu_pre = self.mu[y,x]
        sigma_pre = self.sigma[y,x]#*(1.5**2)# + 0.5**2
        
        #sensor measurement
        z = sensorInvDepth[y,x]
        z_var = sensorVar[y,x]
        h = mu_pre
        
        if z == -1.0:
          #print("no good sensor measurement")
          self.mu[y,x] = mu_pre
          self.sigma[y,x] = sigma_pre*(2.0**2)
          continue
        
        """  
        if abs(z - h) > np.sqrt(sigma_pre):
          #this measurement is probably an outlier
          #
          self.mu[y,x] = mu_pre
          self.sigma[y,x] = sigma_pre*(2.0**2)
          continue
        """
        #kalman gain
        H = 1.0
        S = H*sigma_pre*H + z_var      
        K = sigma_pre*H/S

        #correction
        mu_new = mu_pre + K*(z - h)
        sigma_new = (1 - K*H)*sigma_pre
        
        if mu_new <= 0.0:
          print("bad mu update")
          continue
        if sigma_new <= 0.0:
          print("bad sigma update")
          continue
        
        self.mu[y,x] = mu_new
        self.sigma[y,x] = sigma_new
    
    #self.mu = common.filterInvDepthWithVar(self.mu, self.sigma, self.camera, self.lvl)  
    #self.mu = cv2.bilateralFilter(self.mu,3,0.05,3.0)
    #self.mu = cv2.blur(self.mu,(3,3))
    #self.sigma = cv2.blur(self.sigma,(3,3))

