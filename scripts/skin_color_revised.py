#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np



class Hand_Detection:

	def __init__(self):

		self.lower_bond = np.array([0, 20, 50])
		self.upper_bond = np.array([20, 255, 255])
		self.img = ""



	def skin_mask(self, img):
	
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
		# drawing = np.zeros(skin.shape,np.uint8)
		
		for i in range(len(contours)):
			print "number of contours are"
			print len(contours)
			cnt = contours[i]
			area = cv2.contourArea(cnt)
			print area

			if(area > max_area):
				max_area = area
				cnt_index = i
			else:
				cnt_index = 0

		biggest_cnt = contours[cnt_index]
		# biggest_cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

		return biggest_cnt




	def draw_biggest_contour(self, cnt, img):

		drawing = np.zeros(img.shape)
		# ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		cv2.drawContours(drawing,cnt,-1,(0,255,0),3)

		return drawing




	def find_hull(self, cnt, drawing):

		hull = cv2.convexHull(cnt)
		cv2.drawContours(drawing,[biggest_cnt],0,(0,255,0),2) 
		hull = cv2.convexHull(cnt,returnPoints = False)

		return drawing



	def find_center(self, cnt, img):

		moments = cv2.moments(cnt)
		if moments['m00']!=0:
			cx = int(moments['m10']/moments['m00']) # cx = M10/M00
			cy = int(moments['m01']/moments['m00']) # cy = M01/M00	  
		centr = (cx,cy)       

		"""draw a circle"""
		cv2.circle(img,centr,5,[0,0,255],2)

		return img





if __name__ == '__main__':

	test = Hand_Detection() 

	cap = cv2.VideoCapture(0)
	# while (cap.isOpened()) :
	ret,img = cap.read()
	print "read cam"
	skin = cv2.bitwise_and(img, img, mask = test.skin_mask(img))
	# drawing = test.find_biggest_contour(test.find_contours(skin))
	print "the skin"
	print skin
	contours = test.find_contours(skin)
	print contours
	# cv2.imshow('cnt', contours)
	# cv2.waitKey(2)
	biggest_cnt = test.find_biggest_contour(contours)
	drawing = test.draw_biggest_contour(skin, biggest_cnt)
	
	cv2.imshow("thresh",skin)
	cv2.imshow('output',drawing)
	cv2.imshow('input',img)


	cv2.waitKey(3000)