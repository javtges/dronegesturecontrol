import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import torch
from time import time
import matplotlib.pyplot as plt


# Use Mediapipe to find hand landmarks
# Reorder landmarks
# Find 22nd landmark
# Make correct datasize (100)
# Throw into model
# See what happens

num_classes = 14
num_channels = 66

# model = skeletal_gesture(n_channels=num_channels, n_classes=num_classes)
# model.load_state_dict(torch.load('gesture_pretrained_model_14c_80_3-14.pt'))
# model.eval()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS,30)

sequence_length = 60
input_data = []
value = 0
out = np.zeros(num_classes)

def reorder_landmarks():
    pass


while(True):

  ret, frame = camera.read()
  # image = transform(resized_frame)

  if len(input_data) < sequence_length:
    pass

  if len(input_data) == sequence_length:
    # Make infrence

    # with torch.no_grad():
        # prediction = model(input_data)
        # _, prediction = prediction.max(dim=1)
        # print("predicted {}".format(prediction.tolist()))
    
    pass

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cam.release()
cv2.destroyAllWindows()