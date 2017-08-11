#!/usr/bin/env python
import sys
import rospy
import os

from cacc_msgs.msg import RadarData
from sensor_msgs.msg import CameraInfo,Image

from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import pi,cos,sin,atan2

class RadarVizNode():
  def __init__(self):

    self.bridge = CvBridge()
    
    self.boxWidth = rospy.get_param('~box_width',1.0)
    camPitchDeg = rospy.get_param('~camera_pitch_offset',0.0)
    camYawDeg = rospy.get_param('~camera_yaw_offset',0.0)
    self.camHeight = rospy.get_param('~camera_height_offset',2.0)
    self.camDepth = rospy.get_param('~camera_depth_offset',2.0)
    radarTopic = rospy.get_param('~radar_topic',"/delphi_node/radar_data")
    imageTopic = rospy.get_param('~image_topic',"/axis_decompressed")
    overlayTopic = rospy.get_param('~image_overlay_topic',"/radar_overlay")
    
    self.camPitch = camPitchDeg*pi/180.0
    self.camYaw = camYawDeg*pi/180.0

    radar_sub = rospy.Subscriber(radarTopic,RadarData,self.radarCallback,queue_size=1)
    image_sub = rospy.Subscriber(imageTopic,Image,self.imageCallback,queue_size=1)
    
    self.image_pub = rospy.Publisher(overlayTopic,Image,queue_size=1)

    self.range = np.zeros( (64,1) ) # radar ranges
    self.angle = np.zeros( (64,1) ) # radar angles

    self.Mint = np.matrix((  (941.087953,0.000000,624.481729),
                             (0.000000,942.665887,382.681338),
                             (0.000000,0.000000,1.000000)   ))

    self.imageWidth = 0
    self.imageHeight = 0

    self.xpos = 1.0
    self.ypos = -15.0

  def radarCallback(self,msg):
    for i in range(0,64):
      if ( (msg.status[i]>0) and (msg.track_moving[i]) ):
        self.range[i] = msg.range[i]
        self.angle[i] = msg.azimuth[i]*pi/180
      else:
        self.range[i] = 0.0


  def imageCallback(self,msg):
    # print "image - here"
    # --- Convert to OpenCV image format
    try:
      img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
      print(e)

    shp = img.shape
    self.imageHeight = shp[0]
    self.imageWidth = shp[1]


    ########### Example #############################
    # self.xpos = (self.xpos + 2.5)%75.0

    # r = pow(pow(self.xpos,2) + pow(self.ypos,2),0.5)
    # ang = atan2(self.ypos,self.xpos)

    # boxPoints = self.getBoxPoints(r,ang)

    # self.boxImageOverlay(boxPoints,img)
    #################################################

    # Map angle to pixel location
    for i in range(0,64):
      if (self.range[i]>0):
        boxPoints = self.getBoxPoints(self.range[i],self.angle[i])
        self.boxImageOverlay(boxPoints,img)
        
    img_out = img

    # Convert CV format into ROS format
    image_msg = self.bridge.cv2_to_imgmsg(img_out, "bgr8")

    image_msg.header = msg.header

    self.image_pub.publish(image_msg)

  def boxImageOverlay(self,boxPoints,img):
    for pt1 in boxPoints.T:
        for pt2 in boxPoints.T:
            if pt1 is not pt2:
                px1,py1 = self.getCameraProjection(pt1.T)
                px2,py2 = self.getCameraProjection(pt2.T)
                if ( (px1 is not None) and (px2 is not None) ):
                    cv2.line(img,(px1,py1),(px2,py2),(0,0,255),1)

  def getBoxPoints(self,r,ang):
    r_sp_s = np.matrix( (r*cos(ang),r*sin(ang),0.0) ).T # rpv from radar ("sensor") to point resolved in the sensor frame

    boxPoints = np.concatenate( (r_sp_s,r_sp_s,r_sp_s,r_sp_s) , axis=1)

    dlt = self.boxWidth*0.5

    deltas = np.matrix(( (0.,0.,0.,0.),
                         (1.,1.,-1.,-1.),
                         (1.,-1.,-1.,1.)  ))

    return (boxPoints + deltas)

  def getCameraProjection(self,r_sp_s):

    r_cs_s = np.matrix( (self.camDepth,0.0,self.camHeight) ).T # rpv from camera to radar resolved in the sensor frame

    Ccs = self.rotationMatrix(0.0,self.camPitch,self.camYaw) # rotation matrix from the camera frame to the sensor frame
    Csc = Ccs.T # rotation matrix from the sensor frame to the camera frame

    r_cp_s = r_cs_s + r_sp_s # rpv from camera to point resolved in the sensor frame

    r_cp_c = Csc*r_cp_s # rpv from camera to point resolved in the camera frame

    Rcc = self.rotationMatrix(-pi/2,-pi/2,0.) # rotation matrix from the camera frame to the camera convection frame (z out, y down, x right)

    if (r_cp_c[0]<0):
      return (None,None)

    x_ = self.Mint*Rcc*r_cp_c

    x_im = (x_[0]/x_[2]).astype(int)
    y_im = (x_[1]/x_[2]).astype(int)

    if ( (x_im<0) or (x_im>self.imageWidth) or (y_im<0) or (y_im>self.imageHeight) ):
      return (None,None)
    
    return (x_im,y_im)

  def rotationMatrix(self,rot_x,rot_y,rot_z):
    sx = sin(rot_x)
    sy = sin(rot_y)
    sz = sin(rot_z)
    cx = cos(rot_x)
    cy = cos(rot_y)
    cz = cos(rot_z)

    Rxx = np.matrix((  (1.,0.,0.),(0.,cx,-sx),(0.,sx,cx) ))
    Ryy = np.matrix((  (cy,0.,sy),(0.,1.,0.),(-sy,0.,cy) ))
    Rzz = np.matrix((  (cz,-sz,0.),(sz,cz,0.),(0.,0.,1.) ))

    return Rzz*Ryy*Rxx # Body to nav

def main(args):
  rospy.init_node('radar_viz_node')

  node = RadarVizNode()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    
if __name__ == '__main__':
  main(sys.argv)