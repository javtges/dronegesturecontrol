import pyrealsense2 as rs
import cv2
import numpy as np
import torch
from time import time
import matplotlib.pyplot as plt
import train_infra_pytorch
from torchvision.transforms import *
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from djitellopy import Tello


model = train_infra_pytorch.infra_classifier()
model = nn.DataParallel(model)
model.load_state_dict(torch.load("./network_infra_pytorch_2022-03-17-09-04-30_70.44_.pth"), strict=False)
model.eval()


ker_model = keras.models.load_model('./10_gesture_test_3epoch')


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

colorizer = rs.colorizer()
pipeline.start(config)

profile = pipeline.get_active_profile()
infrared_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
infrared_intrinsics = infrared_profile.get_intrinsics()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

print(infrared_profile)
print(infrared_intrinsics)
print(depth_profile)
print(depth_intrinsics)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# align_to = rs.stream.color
# align = rs.align(align_to)

transform = Compose([transforms.ToTensor(), transforms.CenterCrop((150,150))])

tello = Tello()
tello.connect()
tello.takeoff()

frame_counter = 0

while True:

    frames = pipeline.wait_for_frames()
    # frames = aligned.process(frames)

    infrared_frame_zero = frames.get_infrared_frame(1)
    infrared_frame_one  = frames.get_infrared_frame(2)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    infrared_colormap_zero = np.asanyarray(colorizer.colorize(infrared_frame_zero).get_data())
    infrared_colormap_one = np.asanyarray(colorizer.colorize(infrared_frame_one).get_data())
    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colormap = np.asanyarray(colorizer.colorize(color_frame).get_data())

    depth_frame = np.asanyarray(depth_frame.get_data())
    depth_frame = np.where((depth_frame > clipping_distance), 255, depth_frame)
    depth_frame = cv2.resize(depth_frame, (200, 150), interpolation = cv2.INTER_AREA)
    # print(infrared_colormap_one.shape)
    depth_frame = depth_frame.astype('float32')

    image = torch.from_numpy(depth_frame)

    image = transform(depth_frame)
    # print(image.shape)
    ## Torch only
    # image = image.reshape((1, 1, 150, 150))
    # print(image.shape)

    colormap = cv2.rectangle(colormap, (320 - 240, 240 - 240), (320 + 240, 240 + 240), (0,0,255), 3)
    

    ## Keras only
    image = image.reshape((1, 150, 150, 1))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    prediction = ker_model.predict(image)
    print(prediction)
    m = np.argmax(prediction[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colormap, str(m), (25, 25), font, 1, (0,255,255), 2, cv2.LINE_4)


    ## Torch only
    # with torch.no_grad():

    #     out = model(image)
    #     # print(out)

    #     value, prediction = out.max(dim=1)
    #     print(value, prediction)

    cv2.imshow('RealSense', infrared_colormap_zero)
    images = np.hstack((depth_colormap, colormap))

    cv2.imshow('Other Cameras', images)

    if(m != 9 and frame_counter > 20):
        frame_counter = 0
        if(m == 1):
            tello.move("left", 50)
        elif(m == 5):
            tello.move("right", 50)
        elif(m == 4):
            tello.land()
        elif(m == 0):
            pass
        elif(m==7):
            tello.move("up", 50)
        elif(m==3):
            tello.move("down",50)
        else:
            tello.flip_back()

    frame_counter += 1

    if cv2.waitKey(25) == ord('q'):
        break






pipeline.stop()
cv2.destroyAllWindows()
