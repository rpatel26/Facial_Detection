from Dataset import Dataset
from ComputerVision import ComputerVision

def main():
	print("Main...")
	## (96, 96, 7049)
	images, landmarks = Dataset.load_images()
	cv = ComputerVision()
	img = cv.mark_center(images[:,:,0], landmarks["left_eye_center_x"][0], landmarks["left_eye_center_y"][0])
	cv.show_image(img)

if __name__ == '__main__':
	## Performance criteria: RMSE
	main()	
