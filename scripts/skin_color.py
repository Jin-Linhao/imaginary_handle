#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
import collections



lower = np.array([0, 50, 80], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")





cap = cv2.VideoCapture(0)
if cap.isOpened() == 0 :
    ret,img = cap.read()
else:
	print "failed to initialize camera"
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # ret,thresh1 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    



    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
 
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
 
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask = skinMask)




    bwskin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    newbwskin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, newskinmask  = cv2.threshold(bwskin,150,255,cv2.THRESH_TOZERO_INV)
    contours, hierarchy = cv2.findContours(bwskin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(img.shape,np.uint8)





    max_area=0
   
    for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
    cnt=contours[ci]
    hull = cv2.convexHull(cnt)
    moments = cv2.moments(cnt)
    if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00
              
    centr=(cx,cy)       
    cv2.circle(img,centr,5,[0,0,255],2)       
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
    cv2.drawContours(drawing,[hull],0,(0,0,255),2) 
          
    cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    # if(1):
    #            defects = cv2.convexityDefects(cnt,hull)
    #            mind=0
    #            maxd=0
    #            for i in range(defects.shape[0]):
    #                 s,e,f,d = defects[i,0]
    #                 start = tuple(cnt[s][0])
    #                 end = tuple(cnt[e][0])
    #                 far = tuple(cnt[f][0])
    #                 dist = cv2.pointPolygonTest(cnt,centr,True)
    #                 cv2.line(img,start,end,[0,255,0],2)
                    
    #                 cv2.circle(img,far,5,[0,0,255],-1)
    #            print(i)
    #            i=0
    cv2.imshow("bwthresh1",newskinmask)
    cv2.imshow("bwthresh",newbwskin)           
    cv2.imshow("thresh",skin)
    cv2.imshow('output',drawing)
    cv2.imshow('input',img)
                
    k = cv2.waitKey(10)
    if k == 27:
        break