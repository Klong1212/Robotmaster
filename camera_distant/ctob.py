import cv2

img = cv2.imread(r"camera_distant\sponser_z2.jpg")

cv2.imwrite(r"camera_distant\sponser_z2_binary.jpg", img)