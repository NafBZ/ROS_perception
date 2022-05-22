#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Hands-on Perception Project
# Object detection project.  
# Authors: Umar Muhammad - Nafess Bin Zaman - Ivan Changoluisa
import rospy 
import cv2

import torch
from sensor_msgs.msg import Image as Imagemsg
from cv_bridge import CvBridge, CvBridgeError
from SSD_detector import *


class ObjectDetector: 
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_sub  = rospy.Subscriber('/turtlebot/realsense_d435i/color/image_raw',Imagemsg,self.callback)
    
    def callback(self,data):
        try: 
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        height,width, channels = image.shape
        annotated = inference(image)
        cv2.imshow("Image window", annotated)
        cv2.waitKey(3)
        
    print('Object detector running')


if __name__ == '__main__':
 
    rospy.init_node('perception_test')
    node = ObjectDetector()
    rospy.spin()

