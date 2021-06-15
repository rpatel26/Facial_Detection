import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self):
		pass
		
	@staticmethod
	def load_images(data_path = "./data"):
		assert isinstance(data_path, str)

		result = np.load(data_path + "/face_images.npz")	
		images = result["face_images"]	## (96, 96, 7049)
		landmarks = pd.read_csv(data_path + "/facial_keypoints.csv")	
		return images, landmarks
