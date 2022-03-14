import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import torch
from time import time
import matplotlib.pyplot as plt
from model_skeletal import HandGestureNet


# Use Mediapipe to find hand landmarks
# Reorder landmarks
# Find 22nd landmark
# Make correct datasize (100)
# Throw into model
# See what happens

num_classes = 14
num_channels = 66


## REALSENSE INIT ----------------------------------
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
## --------------------------------------------------------------

## INITALIZE MODEL ----------------------------
model = HandGestureNet(n_channels=num_channels, n_classes=num_classes)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('gesture_pretrained_model_14c_80_3-14.pt'))
else:
    model.load_state_dict(torch.load('gesture_pretrained_model_14c_80_3-14.pt', map_location=torch.device('cpu')))

model.eval()
## ---------------------------------------------


sequence_length = 60
input_data = []
value = 0
out = np.zeros(num_classes)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def reorder_landmarks():
    pass

def get_xyz_from_image(depth_image, x,y):
    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.ppx = 651.0391845703125
    intr.ppy = 354.0467834472656
    intr.fx = 921.5665283203125
    intr.fy = 921.62841796875
    intr.model = rs.distortion.none #"plumb_bob"
    intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

    result = rs.rs2_deproject_pixel_to_point(intr, [x,y], depth_image[y,x])
    return result

def find_coords(img, results):

    lmlist = []
   
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmlist.append([id, cx, cy])
    print(lmlist)

    return lmlist

def find_hand(frame):
        with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.35, min_tracking_confidence=0.35, max_num_hands=1) as hands:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
               
                # print(results.multi_hand_landmarks)

                find_coords(frame, results)

try:
    while(True):

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print("aaa")

        print(color_image.shape)
        print(depth_image.shape)

        find_hand(color_image)

        cv2.imshow("frame", color_image)

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

finally:

    print("done?")
    pipeline.stop()
    cv2.destroyAllWindows()