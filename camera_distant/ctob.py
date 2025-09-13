import cv2

img = cv2.imread(r"camera_distant\sponser_z2.jpg")
h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
h_grass = cv2.GaussianBlur(h, (5, 5), 0)
img_binary = ((h_grass > 30) & (h_grass < 40) & (s > 170))  # ทำเป็น binary สำหรับสีเหลือง
img_binary = img_binary.astype("uint8") * 255
cv2.imwrite(r"camera_distant\sponser_z2_binary.jpg", img_binary)