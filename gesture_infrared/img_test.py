import cv2

image = cv2.imread('../depthGestRecog/fist/video_base_1/s01_g10_011_1.png')
print(image)

while(True):
    cv2.imshow("frame", image)
