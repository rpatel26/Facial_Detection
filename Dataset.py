import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tsp


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
		if len(labels) == 0:
			null_data_rows = landmarks[landmarks.isna().any(axis = 1)].index
		else:
			for label in labels:
				null_data_rows.update(landmarks[landmarks[label].isnull()].index)
			null_data_rows = list(null_data_rows)

		return np.delete(images, null_data_rows, axis = 0), landmarks.drop(null_data_rows).reset_index(drop = True)

	@staticmethod
	def train_test_split(X, Y, train_size = None, test_size = 0.2):
		'''
			Returns: Xtrain, Xtest, Ytrain, Ytest
		'''
		assert isinstance(X, np.ndarray)
		assert isinstance(Y, np.ndarray)
		assert X.shape[0] == Y.shape[0]

		return tsp(X, Y, train_size = train_size, test_size = test_size) 