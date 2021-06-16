from Dataset import Dataset
from ComputerVision import ComputerVision
from Models import Models
import cv2

def main():
	print("Main...")
	## (7049, 96, 96)
	images, landmarks = Dataset.load_images()
	labels = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']
	images, landmarks = Dataset.drop_null_data(images, landmarks, labels = labels)
	Xtrain, Xtest, Ytrain, Ytest = Dataset.train_test_split(images, landmarks[labels].to_numpy())
	print("Xtrain: ", Xtrain.shape)
	print("Ytrain: ", Ytrain.shape)
	print("Xtest: ", Xtest.shape)
	print("Ytest: ", Ytest.shape)



	# cv = ComputerVision()
	# img = cv.mark_center(images[0,:,:], landmarks["left_eye_center_x"][0], landmarks["left_eye_center_y"][0])
	# cv.show_image(img)

	model = Models()
	# keypoint_model = model.build_model()
	# print("model: ", keypoint_model.summary())

	cap = cv2.VideoCapture(0)
	eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


	while True:
		ret, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# model.haarcascade_eye_detector(gray, frame)
		model.haarcascade_face_detector(gray, frame)
		# model.haarcascade_eye_and_face_detector(gray, frame)

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) == ord('q'):
			break



	cap.release()
	cv2.destroyAllWindows()	



if __name__ == '__main__':
	## Performance criteria: RMSE
	main()	
