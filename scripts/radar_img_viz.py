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
    
    self.radPixPerMeterRange = 1.0/5.0
    # camFov = rospy.get_param('/camera_fov',1.0)
    self.pixPerDeg = rospy.get_param('/pixels_per_degree',20.0)
    radarTopic = rospy.get_param('/radar_topic',"/delphi_node/radar_data")
    imageTopic = rospy.get_param('/image_topic',"/axis_decompressed")
    overlayTopic = rospy.get_param('/image_overlay_topic',"/radar_overlay")
    
    radar_sub = rospy.Subscriber(radarTopic,RadarData,self.radar_callback,queue_size=1)
    image_sub = rospy.Subscriber(imageTopic,Image,self.image_callback,queue_size=1)
    
    self.image_pub = rospy.Publisher(overlayTopic,Image,queue_size=1)

    self.range = np.zeros( (64,1) ) # radar ranges
    self.angle = np.zeros( (64,1) ) # radar angles


  def radar_callback(self,msg):
    for i in range(0,64):
      if (msg.status[i]>0):
        self.range[i] = msg.range[i]
        self.angle[i] = msg.azimuth[i]
      else:
        self.range[i] = 0.0


  def image_callback(self,msg):
    # print "image - here"
    # --- Convert to OpenCV image format
    try:
      img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
      print(e)

    shp = img.shape
    imageHeight = shp[0]
    imageWidth = shp[1]

    # Map angle to pixel location
    for i in range(0,64):
      if (self.range[i]>0):
        px = int( self.angle[i] * self.pixPerDeg + imageWidth/2 )
        py = int( imageHeight/2 )
        rad = int(round(self.radPixPerMeterRange*self.range[i]))
        cv2.circle(img,(px,py),rad,(0,0,255))
    
    # for line in square:
    #     imgpts = self.project_line(self.x,line)
        
    img_out = img

    # Convert CV format into ROS format
    image_msg = self.bridge.cv2_to_imgmsg(img_out, "passthrough")

    image_msg.header = msg.header

    self.image_pub.publish(image_msg)


def main(args):
  rospy.init_node('radar_viz_node')

  node = RadarVizNode()
  
  while not rospy.is_shutdown():
    rospy.spin()

if __name__ == '__main__':
  main(sys.argv)