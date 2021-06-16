from Dataset import Dataset
from ComputerVision import ComputerVision
# from Models import Models

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

	# model = Models()
	# keypoint_model = model.build_model()
	# print("model: ", keypoint_model)



if __name__ == '__main__':
	## Performance criteria: RMSE
	main()	
