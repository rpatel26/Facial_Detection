# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, SpatialDropout2D
# from keras.optimizers import SGD
import cv2
import numpy as np

class Models(object):
	"""docstring for Models"""
	def __init__(self):
		self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
		self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')	
		pass

	def build_model(self):
		pass
		
		# model = Sequential()
		# model.add(Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', input_shape=(96,96,1)))
		# model.add(MaxPool2D(pool_size=(2, 2)))
		# model.add(SpatialDropout2D(0.25))
		# model.add(GlobalAveragePooling2D())
		# model.add(Dense(256, activation='relu', kernel_initializer="he_normal"))
		# model.add(Dropout(0.5))
		# model.add(Dense(30, activation='sigmoid'))

		# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		# model.compile(loss='mean_squared_error', optimizer=sgd)	
		# return model

	def haarcascade_eye_detector(self, gray_image, rgb_image):
		assert isinstance(gray_image, np.ndarray)
		assert isinstance(rgb_image, np.ndarray)
		assert len(gray_image.shape) == 2
		assert len(rgb_image.shape) == 3

		face_loc = self.face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5)

		for x, y, dx, dy in face_loc:
			face_img_gray = gray_image[y:y+dy, x:x+dx]
			face_img_rgb = rgb_image[y:y+dy, x:x+dx]
			
			eye_loc = self.eye_cascade.detectMultiScale(face_img_gray, scaleFactor = 1.05, minNeighbors = 5)

			for ey_x, ey_y, ey_dx, ey_dy in eye_loc:
				cv2.rectangle(face_img_rgb, (ey_x, ey_y), (ey_x + ey_dx, ey_y + ey_dy), (0, 255, 0), 3)


	def haarcascade_face_detector(self, gray_image, rgb_image):
		assert isinstance(gray_image, np.ndarray)
		assert isinstance(rgb_image, np.ndarray)
		assert len(gray_image.shape) == 2
		assert len(rgb_image.shape) == 3

		face_loc = self.face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1)
		for x, y, dx, dy in face_loc:
			cv2.rectangle(rgb_image, (x, y), (x + dx, y + dy), (255, 0, 0), 3)	

	def haarcascade_eye_and_face_detector(self, gray_image, rgb_image):
		assert isinstance(gray_image, np.ndarray)
		assert isinstance(rgb_image, np.ndarray)
		assert len(gray_image.shape) == 2
		assert len(rgb_image.shape) == 3

		face_loc = self.face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5)

		for x, y, dx, dy in face_loc:
			cv2.rectangle(rgb_image, (x, y), (x + dx, y + dy), (255, 0, 0), 3)
			face_img_gray = gray_image[y:y+dy, x:x+dx]
			face_img_rgb = rgb_image[y:y+dy, x:x+dx]
			
			eye_loc = self.eye_cascade.detectMultiScale(face_img_gray, scaleFactor = 1.05, minNeighbors = 5)

			for ey_x, ey_y, ey_dx, ey_dy in eye_loc:
				cv2.rectangle(face_img_rgb, (ey_x, ey_y), (ey_x + ey_dx, ey_y + ey_dy), (0, 255, 0), 3)








