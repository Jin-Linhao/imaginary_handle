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


class Hand_Detection:

	def __init__(self):

		# self.bridge = CvBridge()
		self.lower_bond = np.array([0, 20, 50])
		self.upper_bond = np.array([20, 255, 255])
		# self.max_area = 0
		self.img = ""


	def skin_mask(self,  img):

	
		"""HSV thresholding"""

		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		skinMask = cv2.inRange(hsv_img, self.lower_bond, self.upper_bond)
	 

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))		
		skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
		skinMask = cv2.erode(skinMask, kernel, iterations = 2) 
		skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)


		return skinMask



	def find_contours(self, skin):	

		bwskin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
		# newbwskin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, newskinmask  = cv2.threshold(bwskin,150,255,cv2.THRESH_TOZERO_INV)
		contours, hierarchy = cv2.findContours(newskinmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		return contours



	def find_biggest_contour(self, contours):

		max_area = 0
		drawing = np.zeros(skin.shape,np.uint8)

		
		for i in range(len(contours)):
			print len(contours)
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

		return drawing





if __name__ == '__main__':

	test = Hand_Detection()

	cap = cv2.VideoCapture(0)
	while (cap.isOpened()) :
		ret,img = cap.read()
		skin = cv2.bitwise_and(img, img, mask = test.skin_mask(img))
		drawing = test.find_biggest_contour(test.find_contours(skin))
		
		cv2.imshow("thresh",skin)
		cv2.imshow('output',drawing)
		cv2.imshow('input',img)
		# cv2.imshow("images", np.hstack([skin, drawing, img]))
 

		cv2.waitKey(3)