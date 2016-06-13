import numpy as np

import wx
import cv2
import freenect

from gui import BaseLayout

from gestures import HandGestureRecognition



class KinectLayout(BaseLayout):

 	def _create_custom_layout(self):
		pass


 	def _init_custom_layout(self):
	    self.hand_gestures = HandGestureRecognition()


 	def _acquire_frame(self):
		frame, _ = freenect.sync_get_depth()
	   # return success if frame size is valid
	  	if frame is not None:
		   	return (True, frame)
	   	else:
		   	return (False, frame)


	def _process_frame(self, frame):
   # clip max depth to 1023, convert to 8-bit grayscale
   		np.clip(frame, 0, 2**10 – 1, frame)
   		frame >>= 2
   		frame = frame.astype(np.uint8)

   		num_fingers, img_draw = self.hand_gestures.recognize(frame)

   		height, width = frame.shape[:2]
		cv2.circle(img_draw, (width/2, height/2), 3, [255, 102, 0], 2)
		cv2.rectangle(img_draw, (width/3, height/3), (width*2/3, height*2/3), [255, 102, 0], 2)

		cv2.putText(img_draw, str(num_fingers), (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

		return img_draw



class HandGestureRecognition:
  	def __init__(self):
      	 # maximum depth deviation for a pixel to be considered            # within range
       	self.abs_depth_dev = 14

      	 # cut-off angle (deg): everything below this is a            convexity
      	 # point that belongs to two extended fingers
      	self.thresh_deg = 80.0


    def recognize(self, img_gray):
   		segment = self._segment_arm(img_gray)

   		[contours, defects] = self._find_hull_defects(segment)

   		img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
		[num_fingers, img_draw] =    self._detect_num_fingers(contours,
      	defects, img_draw)

      	return (num_fingers, img_draw)


    def _segment_arm(self, frame):
		""" segments the arm region based on depth """
		center_half = 10 # half-width of 21 is 21/2-1
		lowerHeight = self.height/2 – center_half
		upperHeight = self.height/2 + center_half
		lowerWidth = self.width/2 – center_half
		upperWidth = self.width/2 + center_half
		center = frame[lowerHeight:upperHeight,lowerWidth:upperWidth]

		med_val = np.median(center)

		frame = np.where(abs(frame – med_val) <= self.abs_depth_dev,     128, 0).astype(np.uint8)

       	kernel = np.ones((3, 3), np.uint8)
       	frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

       	small_kernel = 3
		frame[ self.height/2-small_kernel: self.height/2+small_kernel, 
			   self.width/2-small_kernel: self.width/2+small_kernel    ] = 128
		
		mask = np.zeros((self.height+2, self.width+2), np.uint8)

		flood = frame.copy()
		cv2.floodFill(flood, mask, (self.width/2, self.height/2), 255, flags=4 | (255 << 8))

		ret, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)


	def _find_hull_defects(self, segment):
   		
   		contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   		
   		max_contour = max(contours, key=cv2.contourArea)
   		

   		defects = cv2.convexityDefects(max_contour, hull)

   		return (cnt,defects)


   	def angle_rad(v1, v2):
   		
   		return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


   	def deg2rad(angle_deg):
   		
   		return angle_deg/180.0*np.pi


   	def _detect_num_fingers(self, contours, defects, img_draw):

   		self.thresh_deg = 80.0

   		if defects is None:
   			return [0, img_draw]
   		if len(defects) <= 2:
   			return [0, img_draw]

   		num_fingers = 1

   		for i in range(defects.shape[0]):
	   	# each defect point is a 4-tuplestart_idx, end_idx,        farthest_idx, _ == defects[i, 0]
	   	start = tuple(contours[start_idx][0])
	   	end = tuple(contours[end_idx][0])
	   	far = tuple(contours[farthest_idx][0])

	   	# draw the hull
	   	cv2.line(img_draw, start, end [0, 255, 0], 2)

	   	# if angle is below a threshold, defect point belongs
		# to two extended fingers
		if angle_rad(np.subtract(start, far), np.subtract(end, far)) < deg2rad(self.thresh_deg):
		    # increment number of fingers
		   num_fingers = num_fingers + 1

		   # draw point as green
		   cv2.circle(img_draw, far, 5, [0, 255, 0], -1)
		else:
		   # draw point as red
		   cv2.circle(img_draw, far, 5, [255, 0, 0], -1)

		return (min(5, num_fingers), img_draw)




def main():
   device = cv2.cv.CV_CAP_OPENNI
   capture = cv2.VideoCapture()
   if not(capture.isOpened()):
	   capture.open(device)

   capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
   capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

	# start graphical user interface
   app = wx.App()
   layout = KinectLayout(None, -1, 'Kinect Hand Gesture Recognition', capture)
   layout.Show(True)
   app.MainLoop()




if __name__ == '__main__':
	main()