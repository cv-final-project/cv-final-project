import os
import shutil

dir = os.listdir()
for file in dir:
	if file.endswith("Mask_Mouth_Chin.jpg"):
		shutil.move(file, "mouth_chin")		
	elif file.endswith("Mask_Nose_Mouth.jpg"):
		shutil.move(file, "mouth_nose")
	elif file.endswith("Mask_Chin.jpg"):
		shutil.move(file, "chin")
	elif file.endswith("Mask.jpg"):
		shutil.move(file, "mouth_nose_chin")
