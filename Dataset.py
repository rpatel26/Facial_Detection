import numpy as np
import pandas as pd


class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self):
		pass
		
	@staticmethod
	def load_images(data_path = "./data"):
		assert isinstance(data_path, str)

		result = np.load(data_path + "/face_images.npz")	
		images = np.moveaxis(result["face_images"], -1, 0)	## (7049, 96, 96)
		landmarks = pd.read_csv(data_path + "/facial_keypoints.csv")	
		return images, landmarks


	@staticmethod
	def drop_null_data(images, landmarks, labels = list()):
		assert isinstance(images, np.ndarray)
		assert isinstance(landmarks, pd.DataFrame)
		assert images.shape[0] == landmarks.shape[0], "num of images and landmarks not same"
		assert set(labels).intersection(landmarks.columns) == set(labels), "labels contain invalid entries"

		null_data_rows = set()
		for label in labels:
			null_data_rows.update(landmarks[landmarks[label].isnull()].index)
		null_data_rows = list(null_data_rows)
		
		return np.delete(images, null_data_rows, axis = 0), landmarks.drop(null_data_rows)