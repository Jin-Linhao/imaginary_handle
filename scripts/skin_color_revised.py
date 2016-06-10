#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np



class Hand_Detection(object):

	def __init__(self):

		self.lower_bond = np.array([0, 20, 50])
		self.upper_bond = np.array([20, 255, 255])
		self.img = ""
		self.max_area = 0


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

		# drawing = np.zeros(skin.shape,np.uint8)
		if len(contours) == 0:

			return None

		for i in range(len(contours)):

			cnt = contours[i]
			area = cv2.contourArea(cnt)

			if(area > self.max_area):
				self.max_area = area
				cnt_index = i
			else:
				cnt_index = 0

		biggest_cnt = contours[cnt_index]
		# biggest_cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

		return biggest_cnt




	def draw_biggest_contour(self, cnt, img):

		drawing = np.zeros(img.shape)
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


	def plot_all(self):

		cap = cv2.VideoCapture(0)
		while (cap.isOpened()) :
			ret,img = cap.read()

			skin = cv2.bitwise_and(img, img, mask = self.skin_mask(img))
			drawing = self.find_biggest_contour(self.find_contours(skin))

			contours = self.find_contours(skin)
			biggest_cnt = self.find_biggest_contour(contours)
			
			if biggest_cnt is None:
				continue

			drawing = self.draw_biggest_contour(biggest_cnt, skin )
			
			cv2.imshow("thresh",skin)
			cv2.imshow('output',drawing)
			cv2.imshow('input',img)

			cv2.waitKey(3)


 

if __name__ == '__main__':

	test = Hand_Detection() 
	test.plot_all()
	