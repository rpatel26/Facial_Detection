import matplotlib.pyplot as plt
import numpy as np
import cv2 


class ComputerVision(object):
	"""docstring for ComputerVision"""
	def __init__(self):
		pass

	def mark_center(self, image, center_x, center_y, color = (255,0,0), radius = 1, thickness = 1):
		assert isinstance(image, np.ndarray)
		assert len(image.shape) == 2	

		if not isinstance(image[0][0], np.float32):
			image = np.float32(image)
		return cv2.circle(image, (int(center_x), int(center_y)), radius, color, thickness)
	

	def show_image(self, image):
		assert isinstance(image, np.ndarray)
		assert len(image.shape) == 2

		plt.imshow(image)
		plt.show()	